"""Streaming discovery, selection, normalization, deduplication, and materialization.

The module intentionally keeps the data plane local: JSONL shards are streamed, exact IDs use
SHA-256, and near-duplicate buckets live in a temporary SQLite database.  Heavy Hugging Face and
FastText dependencies are imported only when a remote source or the production language model is
actually requested.
"""

from __future__ import annotations

import hashlib
import contextlib
import heapq
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 3
DECONTAMINATION_SCHEMA_VERSION = 1
# One below the hard policy ceiling ensures a 100k measurement fixture spans at least three
# shards while remaining within the stated 50k maximum.
MAX_SHARD_ROWS = 49_000
MAX_SHARD_BYTES = 128 * 1024 * 1024
DEFAULT_EXACT_DEDUP_LIMIT = 2_000_000
ALLOWED_SCHEMAS = {"sft", "preference", "cpt", "verified_math", "rlvr"}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CODE_RE = re.compile(r"```.*?```|`[^`]+`", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class IntegrityError(RuntimeError):
    """An authoritative artifact or provenance link failed verification."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_id(text: str) -> str:
    canonical = " ".join(str(text).strip().split())
    return sha256_bytes(canonical.encode("utf-8"))


def strip_code_and_urls(text: str) -> str:
    return _URL_RE.sub(" ", _CODE_RE.sub(" ", text))


class FastTextLanguageIdentifier:
    """Pinned language identifier shared by preparation, evaluation, and rewards.

    ``predictor`` is a small injection point for tests.  Production usage loads one exact model
    file and verifies its SHA-256 before importing fastText.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        expected_hash: Optional[str] = None,
        predictor: Optional[Callable[[str], Tuple[str, float]]] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.expected_hash = expected_hash
        self._predictor = predictor
        self._model = None
        if self.model_path is not None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"language-id model not found: {self.model_path}")
            actual = file_sha256(self.model_path)
            if not expected_hash or actual != expected_hash:
                raise ValueError(
                    f"language-id model hash mismatch: expected {expected_hash!r}, got {actual}"
                )

    @property
    def model_hash(self) -> Optional[str]:
        return self.expected_hash

    def predict(self, text: str) -> Tuple[str, float]:
        cleaned = strip_code_and_urls(text).replace("\n", " ").strip()
        if self._predictor is not None:
            language, confidence = self._predictor(cleaned)
            return str(language).removeprefix("__label__"), float(confidence)
        if self.model_path is None:
            raise RuntimeError("a hashed FastText model or an explicit predictor is required")
        if self._model is None:
            try:
                import fasttext
            except ImportError as exc:
                raise RuntimeError(
                    "FastText language identification requires the data extra"
                ) from exc
            self._model = fasttext.load_model(str(self.model_path))
        labels, probabilities = self._model.predict(cleaned or " ", k=1)
        return labels[0].removeprefix("__label__"), float(probabilities[0])


def language_identifier_from_config(
    data_config: Mapping[str, Any], *, cache_dir: Path
) -> FastTextLanguageIdentifier:
    expected = data_config.get("language_id_sha256")
    configured_path = data_config.get("language_id_model")
    if configured_path:
        return FastTextLanguageIdentifier(Path(configured_path), expected)
    url = data_config.get("language_id_url")
    if not url or not expected:
        raise ValueError("language ID requires a model path or a protected URL and SHA-256")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"fasttext-{expected[:16]}.ftz"
    if not path.exists():
        temporary = path.with_suffix(".download")
        try:
            with (
                urllib.request.urlopen(str(url), timeout=60) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if file_sha256(temporary) != expected:
                raise ValueError("downloaded language-ID model hash mismatch")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return FastTextLanguageIdentifier(path, str(expected))


def _sample_ratio(candidate: Mapping[str, Any]) -> float:
    for key in ("german_sample_ratio", "german_ratio", "de_sample_ratio"):
        value = candidate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return -1.0


def _row_count(candidate: Mapping[str, Any]) -> int:
    value = candidate.get("row_count")
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else -1


def select_sources(
    candidates: Sequence[Mapping[str, Any]],
    *,
    allowed_org: str,
    allowed_licenses: Sequence[str],
    allowed_sources: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Select bounded policy-approved sources deterministically."""
    licenses = {str(value).lower() for value in allowed_licenses}
    source_allowlist = {
        (
            str(source.get("dataset", "")),
            str(source.get("revision", "")),
            str(source.get("license", "")).lower(),
        )
        for source in allowed_sources
    }
    eligible: List[Dict[str, Any]] = []
    for raw in candidates:
        item = dict(raw)
        license_name = str(item.get("license", "")).lower()
        if item.get("training_usable") is not True:
            continue
        if item.get("org", allowed_org) != allowed_org or license_name not in licenses:
            continue
        identity = (
            str(item.get("dataset", "")),
            str(item.get("revision", "")),
            license_name,
        )
        if source_allowlist and identity not in source_allowlist:
            continue
        if item.get("schema") not in {"sft", "preference", "cpt", "verified_math"}:
            continue
        eligible.append(item)
    eligible.sort(
        key=lambda item: (
            -_sample_ratio(item),
            -_row_count(item),
            str(item.get("dataset", "")),
            str(item.get("config", "")),
            str(item.get("split", "")),
        )
    )
    limits = {"sft": 2, "preference": 1, "cpt": 1, "verified_math": 1}
    counts: Counter[str] = Counter()
    selected: List[Dict[str, Any]] = []
    for item in eligible:
        schema = str(item["schema"])
        if counts[schema] >= limits[schema]:
            continue
        counts[schema] += 1
        selected.append(item)
    return selected


def make_selection_artifact(
    discovery_run_id: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    allowed_org: str,
    allowed_licenses: Sequence[str],
    allowed_sources: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    selected = select_sources(
        candidates,
        allowed_org=allowed_org,
        allowed_licenses=allowed_licenses,
        allowed_sources=allowed_sources,
    )
    body = {
        "schema_version": 1,
        "discovery_run_id": discovery_run_id,
        "sources": selected,
        "selection_rule": "de_ratio,row_count,dataset,config,split",
    }
    body["run_id"] = "selection-" + sha256_bytes(canonical_json(body))[:16]
    return {**body, "artifact_hash": sha256_bytes(canonical_json(body))}


def discover_sources(
    *,
    org: str,
    allowed_licenses: Sequence[str],
    sample_size: int = 64,
    dataset_infos: Optional[Iterable[Mapping[str, Any]]] = None,
    language_id: Optional[FastTextLanguageIdentifier] = None,
) -> List[Dict[str, Any]]:
    """Inspect repository metadata and small streamed samples without downloading full datasets."""
    allowed = {value.lower() for value in allowed_licenses}
    if dataset_infos is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("real discovery requires the data extra") from exc

        def hub_infos() -> Iterator[Dict[str, Any]]:
            for info in HfApi().list_datasets(author=org, full=True):
                card = info.card_data
                if hasattr(card, "to_dict"):
                    card = card.to_dict()
                license_name = card.get("license") if isinstance(card, dict) else None
                yield {"dataset": info.id, "license": license_name}

        dataset_infos = hub_infos()
    candidates = []
    for info in dataset_infos:
        dataset = str(info.get("dataset", info.get("id", "")))
        if not dataset:
            continue
        configs = info.get("configs")
        if configs is None:
            try:
                from datasets import get_dataset_config_names

                configs = get_dataset_config_names(dataset, revision=info.get("revision")) or [None]
            except (ImportError, OSError, ValueError) as exc:
                candidates.append(
                    {
                        "dataset": dataset,
                        "org": org,
                        "schema": "unknown",
                        "license": str(info.get("license", "unknown")).lower(),
                        "training_usable": False,
                        "discovery_error": f"could not inspect configs: {type(exc).__name__}: {exc}",
                    }
                )
                continue
        for config_name in configs:
            splits = info.get("splits")
            if isinstance(splits, dict):
                split_values = splits.get(str(config_name), ["train"])
            elif splits is not None:
                split_values = splits
            else:
                try:
                    from datasets import get_dataset_split_names

                    split_values = get_dataset_split_names(
                        dataset, config_name, revision=info.get("revision")
                    ) or ["train"]
                except (ImportError, OSError, ValueError) as exc:
                    candidates.append(
                        {
                            "dataset": dataset,
                            "org": org,
                            "config": config_name,
                            "schema": "unknown",
                            "license": str(info.get("license", "unknown")).lower(),
                            "training_usable": False,
                            "discovery_error": (
                                f"could not inspect splits: {type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                    continue
            for split in split_values:
                samples = []
                if isinstance(info.get("samples"), list):
                    samples = list(info["samples"])[:sample_size]
                elif language_id is not None:
                    try:
                        from datasets import load_dataset
                    except ImportError as exc:
                        raise RuntimeError("sample discovery requires datasets") from exc
                    try:
                        load_kwargs: Dict[str, Any] = {
                            "split": str(split),
                            "streaming": True,
                        }
                        for key in ("revision", "data_files"):
                            if info.get(key) is not None:
                                load_kwargs[key] = info[key]
                        stream = load_dataset(dataset, config_name, **load_kwargs)
                        for index, row in enumerate(stream):
                            if index >= sample_size:
                                break
                            samples.append(dict(row))
                    except (OSError, RuntimeError, ValueError) as exc:
                        candidates.append(
                            {
                                "dataset": dataset,
                                "org": org,
                                "config": config_name,
                                "split": str(split),
                                "schema": "unknown",
                                "license": str(info.get("license", "unknown")).lower(),
                                "training_usable": False,
                                "discovery_error": (
                                    f"could not stream sample: {type(exc).__name__}: {exc}"
                                ),
                            }
                        )
                        continue
                keys = set().union(*(sample.keys() for sample in samples)) if samples else set()
                configured_schema = info.get("schema")
                if configured_schema in ALLOWED_SCHEMAS:
                    schema = str(configured_schema)
                elif {"chosen", "rejected"} <= keys:
                    schema = "preference"
                elif keys & {"instruction", "prompt", "messages", "question"} and keys & {
                    "answer",
                    "response",
                    "output",
                }:
                    schema = "sft"
                else:
                    schema = "cpt"
                german = 0
                checked = 0
                if language_id is not None:
                    for sample in samples:
                        language, _confidence = language_id.predict(_row_text(sample))
                        checked += 1
                        german += int(language == "de")
                ratio = german / checked if checked else float(info.get("german_sample_ratio", -1))
                license_name = str(info.get("license", "unknown")).lower()
                preserved = {
                    key: info[key]
                    for key in (
                        "revision",
                        "data_files",
                        "source_group",
                        "language_filter",
                        "max_rows",
                    )
                    if info.get(key) is not None
                }
                candidates.append(
                    preserved
                    | {
                        "dataset": dataset,
                        "org": str(info.get("org", org)),
                        "config": config_name,
                        "split": str(split),
                        "schema": schema,
                        "license": license_name,
                        "training_usable": license_name in allowed,
                        "german_sample_ratio": ratio,
                        "row_count": info.get("row_count"),
                    }
                )
    return candidates


def verify_hashed_artifact(document: Mapping[str, Any]) -> bool:
    expected = document.get("artifact_hash")
    if not isinstance(expected, str):
        return False
    body = {key: value for key, value in document.items() if key != "artifact_hash"}
    return expected == sha256_bytes(canonical_json(body))


def verify_selection_against_discovery(
    discovery: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    allowed_org: str,
    allowed_licenses: Sequence[str],
    allowed_sources: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Verify that a selection is exactly reproducible from discovery and current policy."""
    if not verify_hashed_artifact(discovery) or not verify_hashed_artifact(selection):
        raise IntegrityError("discovery and selection artifacts must have valid hashes")
    discovery_run_id = discovery.get("run_id")
    if not isinstance(discovery_run_id, str) or not discovery_run_id:
        raise IntegrityError("discovery artifact has no run ID")
    if selection.get("discovery_run_id") != discovery_run_id:
        raise IntegrityError("selection does not reference the supplied discovery run")
    expected = make_selection_artifact(
        discovery_run_id,
        discovery.get("candidates", []),
        allowed_org=allowed_org,
        allowed_licenses=allowed_licenses,
        allowed_sources=allowed_sources,
    )
    if selection.get("run_id") != expected["run_id"]:
        raise IntegrityError("selection run ID is not derived from discovery and current policy")
    if selection.get("sources") != expected["sources"]:
        raise IntegrityError("selected sources differ from discovery and current policy")


def stream_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            yield value


_ROLE_ALIASES = {"human": "user", "gpt": "assistant"}


def _message(item: Mapping[str, Any], *, default_role: Optional[str] = None) -> Dict[str, Any]:
    role = _ROLE_ALIASES.get(
        str(item.get("role", default_role)), str(item.get("role", default_role))
    )
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"invalid conversational role: {role}")
    content = item.get("content", "")
    if not isinstance(content, str):
        raise ValueError("message content must be text")
    for legacy_field in ("function_calls", "functions"):
        value = item.get(legacy_field)
        if value not in (None, "", [], {}):
            raise ValueError(f"unsupported legacy message field: {legacy_field}")
    result: Dict[str, Any] = {"role": role, "content": content}
    if role == "assistant" and "tool_calls" in item:
        calls = item["tool_calls"]
        if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
            raise ValueError("assistant tool_calls must be a list of objects")
        result["tool_calls"] = calls
    if role == "tool":
        tool_call_id = item.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        result["tool_call_id"] = tool_call_id
        if "name" in item:
            if not isinstance(item["name"], str) or not item["name"]:
                raise ValueError("tool message name must be non-empty text")
            result["name"] = item["name"]
    return result


def _conversation(value: Any, role: str) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [_message(item, default_role=role) for item in value]
    raise ValueError(f"expected text or conversational {role} messages")


def _tools(value: Any) -> Optional[List[Dict[str, Any]]]:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise ValueError("tools must be a non-empty list of JSON-schema objects")
    for tool in value:
        if not isinstance(tool.get("type", "function"), str):
            raise ValueError("tool type must be text")
        function = tool.get("function", tool)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("each tool requires a function name")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("tool parameters must be a JSON schema object")
    return [dict(tool) for tool in value]


def infer_schema(row: Mapping[str, Any], configured_schema: Optional[str] = None) -> str:
    """Classify specific structured rows before general conversational SFT rows."""
    if "chosen" in row and "rejected" in row:
        return "preference"
    if configured_schema == "verified_math":
        return "verified_math"
    if "messages" in row or (
        any(key in row for key in ("instruction", "prompt", "question"))
        and any(key in row for key in ("answer", "response", "output"))
    ):
        return "sft"
    return configured_schema or "cpt"


def _require_user_and_assistant(messages: Sequence[Mapping[str, Any]]) -> None:
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("conversation requires a user turn")
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("conversation requires an assistant turn")


def _completion(value: Any) -> List[Dict[str, Any]]:
    messages = _conversation(value, "assistant")
    if not any(
        message.get("role") == "assistant" and message.get("content", "").strip()
        for message in messages
    ):
        raise ValueError("preference completion requires non-empty assistant output")
    return messages


def normalize_row(row: Mapping[str, Any], schema: str, source: Mapping[str, Any]) -> Dict[str, Any]:
    schema = infer_schema(row, schema)
    if schema == "sft":
        if row.get("messages") is not None:
            messages = _conversation(row["messages"], "user")
        else:
            prompt = row.get("prompt", row.get("instruction", row.get("question")))
            answer = row.get("response", row.get("output", row.get("answer")))
            messages = _conversation(prompt, "user") + _conversation(answer, "assistant")
        _require_user_and_assistant(messages)
        normalized = {"type": "sft", "messages": messages, "tools": _tools(row.get("tools"))}
        text = json.dumps(
            {"messages": messages, "tools": normalized["tools"]}, ensure_ascii=False, sort_keys=True
        )
    elif schema == "preference":
        prompt_value = row.get("prompt", row.get("messages", row.get("instruction")))
        chosen_value, rejected_value = row.get("chosen"), row.get("rejected")
        if (
            prompt_value is None
            and isinstance(chosen_value, list)
            and isinstance(rejected_value, list)
        ):
            chosen_full = _conversation(chosen_value, "assistant")
            rejected_full = _conversation(rejected_value, "assistant")
            common = 0
            for chosen_message, rejected_message in zip(chosen_full, rejected_full):
                if chosen_message != rejected_message:
                    break
                common += 1
            if common == 0:
                raise ValueError("full-conversation preferences require a common prompt prefix")
            prompt = chosen_full[:common]
            chosen_value = chosen_full[common:]
            rejected_value = rejected_full[common:]
        else:
            if prompt_value is None:
                raise ValueError("preference row requires prompt or messages")
            prompt = _conversation(prompt_value, "user")
            if isinstance(chosen_value, list) and isinstance(rejected_value, list):
                chosen_full = _conversation(chosen_value, "assistant")
                rejected_full = _conversation(rejected_value, "assistant")
                if chosen_full[: len(prompt)] == prompt and rejected_full[: len(prompt)] == prompt:
                    chosen_value = chosen_full[len(prompt) :]
                    rejected_value = rejected_full[len(prompt) :]
        if not any(message.get("role") == "user" for message in prompt):
            raise ValueError("preference prompt requires a user turn")
        if prompt[-1].get("role") != "user":
            raise ValueError("preference prompt must end with a user turn")
        chosen = _completion(chosen_value)
        rejected = _completion(rejected_value)
        chosen_content = [
            message.get("content", "").strip()
            for message in chosen
            if message.get("role") == "assistant"
        ]
        rejected_content = [
            message.get("content", "").strip()
            for message in rejected
            if message.get("role") == "assistant"
        ]
        if chosen_content == rejected_content:
            raise ValueError("chosen and rejected assistant continuations must differ")
        normalized = {
            "type": "preference",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "tools": _tools(row.get("tools")),
        }
        text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    elif schema == "cpt":
        text = str(row.get("text", row.get("content", ""))).strip()
        if not text:
            raise ValueError("empty CPT row")
        normalized = {"type": "cpt", "text": text}
    elif schema == "verified_math":
        prompt_value = row.get("prompt", row.get("messages", row.get("question")))
        prompt = _conversation(prompt_value, "user")
        if not prompt or prompt[-1].get("role") != "user":
            raise ValueError("verified_math prompt must end in a user turn")
        solution = row.get("solution", row.get("answer"))
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("verified_math solution must be non-empty text")
        normalized = {"type": "verified_math", "prompt": prompt, "solution": solution.strip()}
        text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    elif schema == "rlvr":
        normalized = dict(row)
        normalized["prompt"] = _conversation(row.get("prompt"), "user")
        text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    else:
        raise ValueError(f"unsupported schema: {schema}")
    normalized.update(
        {
            "schema": schema,
            "source_group": str(source.get("source_group", source.get("dataset", "unknown"))),
            "license": source.get("license"),
            "content_id": content_id(text),
        }
    )
    return normalized


def _shingles(text: str, width: int = 5) -> set[str]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _minhash_signature(text: str, permutations: int = 16) -> Tuple[int, ...]:
    shingles = _shingles(text)
    if not shingles:
        return tuple(0 for _ in range(permutations))
    signature = []
    for seed in range(permutations):
        signature.append(
            min(
                int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16)
                for value in shingles
            )
        )
    return tuple(signature)


class NearDuplicateIndex:
    """Deterministic fixed-band MinHash index backed by local SQLite."""

    def __init__(self, path: Path, bands: int = 4, rows_per_band: int = 4):
        self.path = Path(path)
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=MEMORY")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(
            "CREATE TABLE buckets (band INTEGER, bucket TEXT, content_id TEXT, PRIMARY KEY "
            "(band, bucket, content_id))"
        )
        self.connection.execute("CREATE INDEX bucket_lookup ON buckets(band, bucket)")
        self.pending = 0

    def is_duplicate_and_add(self, cid: str, text: str) -> bool:
        signature = _minhash_signature(text, self.bands * self.rows_per_band)
        buckets = []
        for band in range(self.bands):
            start = band * self.rows_per_band
            bucket = sha256_bytes(canonical_json(signature[start : start + self.rows_per_band]))
            buckets.append((band, bucket))
        duplicate = any(
            self.connection.execute(
                "SELECT 1 FROM buckets WHERE band=? AND bucket=? LIMIT 1", pair
            ).fetchone()
            is not None
            for pair in buckets
        )
        if not duplicate:
            self.connection.executemany(
                "INSERT INTO buckets(band,bucket,content_id) VALUES(?,?,?)",
                [(band, bucket, cid) for band, bucket in buckets],
            )
            self.pending += 1
            if self.pending >= 1_000:
                self.connection.commit()
                self.pending = 0
        return duplicate

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _row_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _dedup_text(row: Mapping[str, Any]) -> str:
    semantic = {
        key: row[key]
        for key in ("messages", "prompt", "chosen", "rejected", "text", "solution", "ground_truth")
        if key in row
    }
    return json.dumps(semantic, ensure_ascii=False, sort_keys=True)


def _completion_language_texts(row: Mapping[str, Any]) -> Iterator[str]:
    """Yield model outputs that must independently satisfy the row language policy."""
    schema = row.get("schema")
    if schema == "sft":
        groups = (row.get("messages", []),)
    elif schema == "preference":
        groups = (row.get("chosen", []), row.get("rejected", []))
    elif schema in {"verified_math", "rlvr"}:
        for key in ("solution", "ground_truth"):
            value = row.get(key)
            if isinstance(value, str):
                yield value
        return
    else:
        return
    for messages in groups:
        if not isinstance(messages, list):
            continue
        for message in messages:
            if (
                isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
            ):
                yield str(message["content"])


def _requires_independent_language_check(text: str) -> bool:
    """Avoid applying FastText to labels, numeric answers, URLs, or code-only completions."""
    cleaned = strip_code_and_urls(text)
    tokens = _TOKEN_RE.findall(cleaned)
    alphabetic = sum(any(character.isalpha() for character in token) for token in tokens)
    return alphabetic >= 4 and len(cleaned.strip()) >= 20


def _completion_language_is_allowed(
    row: Mapping[str, Any],
    language_id: FastTextLanguageIdentifier,
    *,
    requested_language: str,
    minimum_confidence: float,
) -> bool:
    for text in _completion_language_texts(row):
        if not _requires_independent_language_check(text):
            continue
        language, confidence = language_id.predict(text)
        if language != requested_language or confidence < minimum_confidence:
            return False
    return True


@dataclass
class _ShardWriter:
    directory: Path
    schema: str
    group: str
    split: str
    max_rows: int
    max_bytes: int
    index: int = 0
    rows: int = 0
    size: int = 0
    handle: Any = None
    path: Optional[Path] = None

    def _open(self) -> None:
        self.index += 1
        safe_group = re.sub(r"[^A-Za-z0-9._-]+", "-", self.group).strip("-") or "source"
        self.path = self.directory / (
            f"{self.schema}-{self.split}-{safe_group}-{self.index:05d}.jsonl"
        )
        self.handle = self.path.open("w", encoding="utf-8")
        self.rows = self.size = 0

    def write(self, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        byte_count = len(encoded.encode("utf-8"))
        completed = None
        if self.handle is None:
            self._open()
        elif self.rows and (self.rows >= self.max_rows or self.size + byte_count > self.max_bytes):
            completed = self.close()
            self._open()
        self.handle.write(encoded)
        self.rows += 1
        self.size += byte_count
        return completed

    def close(self) -> Optional[Dict[str, Any]]:
        if self.handle is None or self.path is None:
            return None
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        result = {
            "path": self.path.name,
            "schema": self.schema,
            "split": self.split,
            "role": (
                f"{self.schema}_validation_shard"
                if self.split == "validation"
                else f"{self.schema}_shard"
            ),
            "source_group": self.group,
            "rows": self.rows,
            "bytes": self.size,
            "sha256": file_sha256(self.path),
        }
        self.handle = self.path = None
        return result


def _source_rows(source: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    if source.get("rows") is not None:
        for row in source["rows"]:
            if not isinstance(row, dict):
                raise ValueError("source rows must be objects")
            yield row
        return
    if source.get("path"):
        yield from stream_jsonl(Path(str(source["path"])))
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("remote materialization requires the data extra") from exc
    load_kwargs: Dict[str, Any] = {
        "split": str(source.get("split", "train")),
        "streaming": True,
    }
    for key in ("revision", "data_files"):
        if source.get(key) is not None:
            load_kwargs[key] = source[key]
    dataset = load_dataset(str(source["dataset"]), source.get("config"), **load_kwargs)
    for row in dataset:
        yield dict(row)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile without extra dependencies."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile quantile must be between zero and one")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    ordered = [float(value) for value in values]
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total": total,
        "min": min(ordered) if ordered else 0.0,
        "max": max(ordered) if ordered else 0.0,
        "mean": total / len(ordered) if ordered else 0.0,
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
    }


def _template_encoding(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    assistant_mask: bool = False,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], Optional[List[int]]]:
    kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": True,
    }
    if tools is not None:
        kwargs["tools"] = list(tools)
    if assistant_mask:
        kwargs["return_assistant_tokens_mask"] = True
    try:
        encoded = tokenizer.apply_chat_template(list(messages), **kwargs)
    except (TypeError, ValueError, AttributeError) as exc:
        detail = " with tools" if tools is not None else ""
        raise ValueError(f"chat template cannot render conversation{detail}: {exc}") from exc
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids")
        mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    else:
        ids, mask = encoded, None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise ValueError("chat template did not return input_ids")
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if mask and isinstance(mask[0], list):
        mask = mask[0]
    if assistant_mask and (not isinstance(mask, list) or len(mask) != len(ids)):
        raise ValueError("chat template did not return a valid assistant token mask")
    return [int(value) for value in ids], [
        int(value) for value in mask
    ] if mask is not None else None


def token_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    context_length: int,
    max_prompt_length: int,
    max_completion_length: int,
) -> Dict[str, Any]:
    """Measure canonical rows with the tokenizer and exact chat template used for training."""
    stats: Dict[str, Any] = {}
    by_schema: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_schema.setdefault(str(row["schema"]), []).append(row)

    sft_rows = by_schema.get("sft", [])
    sft_lengths: List[int] = []
    assistant_counts: List[int] = []
    truncated_assistant_counts: List[int] = []
    sft_truncated = sft_empty = sft_lost = sft_last_lost = 0
    for row in sft_rows:
        ids, mask = _template_encoding(
            tokenizer, row["messages"], tools=row.get("tools"), assistant_mask=True
        )
        assert mask is not None
        supervised = sum(mask)
        truncated_supervised = sum(mask[:context_length])
        last_assistant = max(
            index
            for index, message in enumerate(row["messages"])
            if message.get("role") == "assistant"
        )
        before_ids, _ = _template_encoding(
            tokenizer,
            row["messages"][:last_assistant],
            tools=row.get("tools"),
        )
        through_ids, through_mask = _template_encoding(
            tokenizer,
            row["messages"][: last_assistant + 1],
            tools=row.get("tools"),
            assistant_mask=True,
        )
        assert through_mask is not None
        last_mask = through_mask[len(before_ids) : len(through_ids)]
        visible_last_mask = through_mask[len(before_ids) : min(len(through_ids), context_length)]
        sft_lengths.append(len(ids))
        assistant_counts.append(supervised)
        truncated_assistant_counts.append(truncated_supervised)
        sft_empty += int(supervised == 0)
        sft_truncated += int(len(ids) > context_length)
        sft_lost += int(supervised > 0 and truncated_supervised == 0)
        sft_last_lost += int(sum(last_mask) > 0 and sum(visible_last_mask) == 0)
    sft_total = sum(min(length, context_length) for length in sft_lengths)
    stats["sft"] = {
        "sequence_tokens": distribution(sft_lengths),
        "assistant_tokens": distribution(assistant_counts),
        "supervised_token_fraction": (
            sum(truncated_assistant_counts) / sft_total if sft_total else 0.0
        ),
        "examples_without_assistant_tokens": sft_empty,
        "truncated_without_supervision": sft_lost,
        "last_assistant_turn_fully_truncated": sft_last_lost,
        "truncation_fraction": sft_truncated / len(sft_rows) if sft_rows else 0.0,
    }

    pref_rows = by_schema.get("preference", [])
    prompt_lengths: List[int] = []
    chosen_lengths: List[int] = []
    rejected_lengths: List[int] = []
    chosen_totals: List[int] = []
    rejected_totals: List[int] = []
    ratios: List[float] = []
    prompt_clipped = completion_clipped = context_clipped = 0
    for row in pref_rows:
        prompt_ids, _ = _template_encoding(
            tokenizer,
            row["prompt"],
            tools=row.get("tools"),
            add_generation_prompt=True,
        )
        chosen_ids, _ = _template_encoding(tokenizer, row["chosen"])
        rejected_ids, _ = _template_encoding(tokenizer, row["rejected"])
        prompt_len, chosen_len, rejected_len = len(prompt_ids), len(chosen_ids), len(rejected_ids)
        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)
        chosen_totals.append(prompt_len + chosen_len)
        rejected_totals.append(prompt_len + rejected_len)
        ratios.append(max(chosen_len, rejected_len) / max(1, min(chosen_len, rejected_len)))
        prompt_clipped += int(prompt_len > max_prompt_length)
        completion_clipped += int(
            chosen_len > max_completion_length or rejected_len > max_completion_length
        )
        context_clipped += int(prompt_len + max(chosen_len, rejected_len) > context_length)
    stats["preference"] = {
        "prompt_tokens": distribution(prompt_lengths),
        "chosen_completion_tokens": distribution(chosen_lengths),
        "rejected_completion_tokens": distribution(rejected_lengths),
        "chosen_sequence_tokens": distribution(chosen_totals),
        "rejected_sequence_tokens": distribution(rejected_totals),
        "chosen_rejected_length_ratio": distribution(ratios),
        "prompt_truncation_fraction": prompt_clipped / len(pref_rows) if pref_rows else 0.0,
        "completion_truncation_fraction": (
            completion_clipped / len(pref_rows) if pref_rows else 0.0
        ),
        "context_truncation_fraction": context_clipped / len(pref_rows) if pref_rows else 0.0,
    }

    cpt_lengths = [
        len(tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"])
        for row in by_schema.get("cpt", [])
    ]
    stats["cpt"] = {
        "sequence_tokens": distribution(cpt_lengths),
        "truncation_fraction": (
            sum(length > context_length for length in cpt_lengths) / len(cpt_lengths)
            if cpt_lengths
            else 0.0
        ),
    }

    math_prompt: List[int] = []
    math_solution: List[int] = []
    unparseable_gold = 0
    math_rows = by_schema.get("verified_math", [])
    if math_rows:
        from .verified_rl import parse_gold_solution
    for row in math_rows:
        ids, _ = _template_encoding(tokenizer, row["prompt"], add_generation_prompt=True)
        math_prompt.append(len(ids))
        math_solution.append(
            len(tokenizer(str(row["solution"]), add_special_tokens=False)["input_ids"])
        )
        try:
            parse_gold_solution(str(row["solution"]))
        except (RuntimeError, ValueError):
            unparseable_gold += 1
    stats["verified_math"] = {
        "prompt_tokens": distribution(math_prompt),
        "solution_tokens": distribution(math_solution),
        "empty_or_unparseable_gold_fraction": (
            unparseable_gold / len(math_rows) if math_rows else 0.0
        ),
    }
    return stats


def _sampling_key(seed: int, source: Mapping[str, Any], cid: str) -> int:
    fields = (
        "sampling-v1",
        str(seed),
        str(source.get("dataset", source.get("path", "inline"))),
        str(source.get("revision", "")),
        str(source.get("config", "")),
        str(source.get("split", "train")),
        cid,
    )
    return int(sha256_bytes("\0".join(fields).encode("utf-8")), 16)


def _split_key(seed: int, cid: str) -> str:
    return sha256_bytes(f"validation-v1\0{seed}\0{cid}".encode("utf-8"))


def _bounded_sample(
    candidates: Iterable[Tuple[int, Dict[str, Any]]], limit: Optional[int]
) -> List[Tuple[int, Dict[str, Any]]]:
    if limit is None:
        return sorted(candidates, key=lambda item: (item[0], item[1]["content_id"]))
    if limit <= 0:
        return []
    heap: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for ordinal, (priority, row) in enumerate(candidates):
        tie = int(row["content_id"], 16)
        entry = (-priority, -tie, -ordinal, row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return sorted(
        [(-priority, row) for priority, _tie, _ordinal, row in heap],
        key=lambda item: (item[0], item[1]["content_id"]),
    )


def materialize_streaming(
    sources: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    language_id: Optional[FastTextLanguageIdentifier] = None,
    min_german_confidence: float = 0.0,
    max_rows: int = MAX_SHARD_ROWS,
    max_bytes: int = MAX_SHARD_BYTES,
    exact_dedup_limit: int = DEFAULT_EXACT_DEDUP_LIMIT,
    decontamination_hash: Optional[str] = None,
    decontamination_corpus: Optional[Mapping[str, Any]] = None,
    policy_hash: Optional[str] = None,
    discovery_run_id: Optional[str] = None,
    selection_run_id: Optional[str] = None,
    seed: int = 17,
    max_rows_per_source: Optional[int] = None,
    global_max_rows: Optional[int] = None,
    validation_fraction: Optional[Mapping[str, float]] = None,
    tokenizer: Any = None,
    context_length: int = 16_384,
    max_prompt_length: int = 4_096,
    max_completion_length: int = 2_048,
) -> Dict[str, Any]:
    """Scan pinned streams, hash-sample them, split them, and publish bounded shards."""
    if decontamination_corpus is not None:
        if not verify_hashed_artifact(decontamination_corpus):
            raise ValueError("decontamination corpus artifact hash is invalid")
        corpus_hash = str(decontamination_corpus["artifact_hash"])
        if decontamination_hash is not None and decontamination_hash != corpus_hash:
            raise ValueError("configured decontamination hash does not match the corpus")
        if policy_hash is not None and decontamination_corpus.get("policy_hash") != policy_hash:
            raise ValueError("decontamination corpus policy hash is stale")
        decontamination_hash = corpus_hash
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-staging-", dir=out_dir.parent))
    near_path = staging / "near-dedup.sqlite3"
    near = NearDuplicateIndex(near_path)
    exact: set[str] = set()
    writers: Dict[Tuple[str, str, str], _ShardWriter] = {}
    shards: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    fractions = {
        "sft": 0.0,
        "cpt": 0.0,
        "preference": 0.0,
        "verified_math": 0.0,
        **dict(validation_fraction or {}),
    }
    for schema, fraction in fractions.items():
        if schema not in {"sft", "cpt", "preference", "verified_math", "rlvr"}:
            raise ValueError(f"unknown validation fraction schema: {schema}")
        if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) < 0.5:
            raise ValueError(f"validation fraction for {schema} must satisfy 0.0 <= value < 0.5")
    try:
        source_samples: List[Tuple[int, Dict[str, Any]]] = []
        sampled_per_source: List[Dict[str, Any]] = []
        for source in sorted(
            sources,
            key=lambda item: (
                str(item.get("dataset", item.get("path", "inline"))),
                str(item.get("revision", "")),
                str(item.get("config", "")),
                str(item.get("split", "train")),
                str(item.get("schema", "")),
            ),
        ):
            schema = str(source.get("schema"))
            if schema not in ALLOWED_SCHEMAS:
                raise ValueError(f"unsupported source schema: {schema}")

            def normalized_candidates() -> Iterator[Tuple[int, Dict[str, Any]]]:
                for raw in _source_rows(source):
                    counts["rows_scanned"] += 1
                    counts["seen"] += 1  # retained for manifest compatibility
                    try:
                        row = normalize_row(raw, schema, source)
                    except (TypeError, ValueError, KeyError) as exc:
                        counts["invalid"] += 1
                        rejection_reasons[f"structural:{type(exc).__name__}:{exc}"] += 1
                        continue
                    counts["rows_structurally_valid"] += 1
                    requested_language = source.get("language_filter")
                    if requested_language is not None:
                        if language_id is None:
                            raise ValueError(
                                "source language_filter requires a configured language identifier"
                            )
                        text = _dedup_text(row)
                        language, confidence = language_id.predict(text)
                        languages[language] += 1
                        row["language"] = {
                            "id": language,
                            "confidence": confidence,
                            "model_hash": language_id.model_hash,
                        }
                        if (
                            language != str(requested_language)
                            or confidence < min_german_confidence
                        ):
                            counts["language_rejected"] += 1
                            rejection_reasons["language"] += 1
                            continue
                        if not _completion_language_is_allowed(
                            row,
                            language_id,
                            requested_language=str(requested_language),
                            minimum_confidence=min_german_confidence,
                        ):
                            counts["language_rejected"] += 1
                            rejection_reasons["assistant_language"] += 1
                            continue
                    yield _sampling_key(seed, source, row["content_id"]), row

            source_limit = source.get("max_rows", max_rows_per_source)
            selected = _bounded_sample(
                normalized_candidates(),
                int(source_limit) if source_limit is not None else None,
            )
            counts["rows_sampled_per_source"] += len(selected)
            sampled_per_source.append(
                {
                    "dataset": str(source.get("dataset", source.get("path", "inline"))),
                    "revision": str(source.get("revision", "")),
                    "config": str(source.get("config", "")),
                    "split": str(source.get("split", "train")),
                    "schema": schema,
                    "rows_sampled": len(selected),
                }
            )
            source_samples.extend(selected)

        selected_rows = _bounded_sample(source_samples, global_max_rows)
        counts["rows_sampled"] = len(selected_rows)
        trainable_rows: List[Dict[str, Any]] = []
        for _priority, row in selected_rows:
            cid = row["content_id"]
            if cid in exact:
                counts["exact_duplicates"] += 1
                rejection_reasons["exact_duplicate"] += 1
                continue
            if len(exact) >= exact_dedup_limit:
                raise MemoryError(f"exact dedup policy limit exceeded: {exact_dedup_limit}")
            exact.add(cid)
            text = _dedup_text(row)
            if near.is_duplicate_and_add(cid, text):
                counts["near_duplicates"] += 1
                rejection_reasons["near_duplicate"] += 1
                continue
            if decontamination_corpus is not None and not decontaminate(
                row, decontamination_corpus
            ):
                counts["leakage_rejected"] += 1
                rejection_reasons["leakage"] += 1
                continue
            if language_id is not None and "language" not in row:
                language, confidence = language_id.predict(text)
                languages[language] += 1
                row["language"] = {
                    "id": language,
                    "confidence": confidence,
                    "model_hash": language_id.model_hash,
                }
                if language != "de" or confidence < min_german_confidence:
                    counts["language_rejected"] += 1
                    rejection_reasons["language"] += 1
                    continue
                if not _completion_language_is_allowed(
                    row,
                    language_id,
                    requested_language="de",
                    minimum_confidence=min_german_confidence,
                ):
                    counts["language_rejected"] += 1
                    rejection_reasons["assistant_language"] += 1
                    continue
            trainable_rows.append(row)
        counts["rows_trainable"] = len(trainable_rows)
        counts["written"] = len(trainable_rows)

        split_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        split_statistics: Dict[str, Dict[str, int]] = {}
        for schema in ("sft", "cpt", "preference", "verified_math", "rlvr"):
            rows = sorted(
                (row for row in trainable_rows if row["schema"] == schema),
                key=lambda row: (_split_key(seed, row["content_id"]), row["content_id"]),
            )
            fraction = float(fractions.get(schema, 0.0))
            if fraction > 0 and len(rows) == 1:
                raise ValueError(
                    f"{schema} validation is enabled but only one trainable row remains"
                )
            validation_count = max(1, int(len(rows) * fraction)) if fraction and rows else 0
            validation = rows[:validation_count]
            train = rows[validation_count:]
            if {row["content_id"] for row in train} & {row["content_id"] for row in validation}:
                raise RuntimeError(f"{schema} train/validation split overlap")
            split_rows[(schema, "train")] = train
            split_rows[(schema, "validation")] = validation
            split_statistics[schema] = {
                "train": len(train),
                "validation": len(validation),
            }

        for (schema, split), rows in sorted(split_rows.items()):
            for row in sorted(
                rows,
                key=lambda item: (
                    str(item.get("source_group", "unknown")),
                    item["content_id"],
                ),
            ):
                group = str(row.get("source_group", "unknown"))
                writer = writers.setdefault(
                    (schema, split, group),
                    _ShardWriter(staging, schema, group, split, max_rows, max_bytes),
                )
                completed = writer.write(row)
                if completed:
                    shards.append(completed)
        for writer in writers.values():
            completed = writer.close()
            if completed:
                shards.append(completed)
        near.close()
        near_path.unlink()
        shards.sort(key=lambda item: item["path"])

        measured = (
            token_statistics(
                trainable_rows,
                tokenizer=tokenizer,
                context_length=context_length,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
            )
            if tokenizer is not None
            else {}
        )
        source_documents = [dict(source) | {"rows": None} for source in sources]
        source_documents.sort(
            key=lambda item: (
                str(item.get("dataset", item.get("path", "inline"))),
                str(item.get("revision", "")),
                str(item.get("config", "")),
                str(item.get("split", "train")),
                str(item.get("schema", "")),
            )
        )
        manifest_body = {
            "schema_version": SCHEMA_VERSION,
            "status": "trainable",
            "mode": "real",
            "sampling": {
                "algorithm": "sha256-smallest-v1",
                "seed": seed,
                "max_rows_per_source": max_rows_per_source,
                "max_rows": global_max_rows,
            },
            "sources": source_documents,
            "shards": shards,
            "row_counts": dict(sorted(counts.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "split_statistics": split_statistics,
            "token_statistics": measured,
            "language_counts": dict(sorted(languages.items())),
            "decontamination_hash": decontamination_hash,
            "policy_hash": policy_hash,
            "discovery_run_id": discovery_run_id,
            "selection_run_id": selection_run_id,
        }
        manifest = {**manifest_body, "artifact_hash": sha256_bytes(canonical_json(manifest_body))}
        if decontamination_corpus is not None:
            (staging / "decontamination.json").write_text(
                json.dumps(decontamination_corpus, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (staging / "leakage_report.json").write_text(
            json.dumps(
                {
                    "status": "verified_clean"
                    if decontamination_corpus is not None
                    else "not_checked",
                    "decontamination_hash": decontamination_hash,
                    "overlap_hits": counts.get("leakage_rejected", 0),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (staging / "quality_report.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "counts": dict(sorted(counts.items())),
                    "sampled_per_source": sampled_per_source,
                    "rejection_reasons": dict(sorted(rejection_reasons.items())),
                    "languages": dict(sorted(languages.items())),
                    "split_statistics": split_statistics,
                    "token_statistics": measured,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Discovery and deterministic selection are part of the same artifact directory but are
        # inputs, not regenerated outputs. Preserve them across the atomic directory swap.
        if out_dir.exists():
            for name in ("discovery.json", "selection.json", "decontamination.json"):
                if name == "decontamination.json" and decontamination_corpus is not None:
                    continue
                source = out_dir / name
                if source.is_file():
                    shutil.copy2(source, staging / name)
        if out_dir.exists():
            backup = out_dir.with_name(f".{out_dir.name}-previous")
            if backup.exists():
                shutil.rmtree(backup)
            out_dir.rename(backup)
            staging.rename(out_dir)
            shutil.rmtree(backup)
        else:
            staging.rename(out_dir)
        return manifest
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            near.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def iter_manifest_rows(
    manifest_path: Path,
    *,
    expected_decontamination_hash: Optional[str] = None,
    schema: Optional[str] = None,
    split: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not verify_hashed_artifact(manifest):
        raise ValueError("data manifest artifact hash is invalid")
    if expected_decontamination_hash is not None and (
        manifest.get("decontamination_hash") != expected_decontamination_hash
    ):
        raise ValueError("data manifest has a stale decontamination hash")
    for shard in manifest.get("shards", []):
        if schema is not None and shard.get("schema") != schema:
            continue
        if split is not None and shard.get("split", "train") != split:
            continue
        path = manifest_path.parent / shard["path"]
        if file_sha256(path) != shard["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        yield from stream_jsonl(path)


def load_manifest_rows(
    manifest: Mapping[str, Any] | Path,
    kind: str,
    *,
    split: str,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load one verified manifest role explicitly; never derive a trainer-side split."""
    if split not in {"train", "validation"}:
        raise ValueError("manifest split must be train or validation")
    if isinstance(manifest, Path):
        path = manifest
        document = verify_data_manifest(path)
    else:
        document = dict(manifest)
        path = Path(root or ".") / "manifest.json"
        if not verify_hashed_artifact(document):
            raise IntegrityError("data manifest artifact hash is invalid")
    schema = "sft" if kind in {"sft", "specialist"} else kind
    rows: List[Dict[str, Any]] = []
    for shard in document.get("shards", []):
        if shard.get("schema") != schema or shard.get("split", "train") != split:
            continue
        shard_path = (
            path.parent / shard["path"]
            if isinstance(manifest, Path)
            else Path(root or ".") / shard["path"]
        )
        if file_sha256(shard_path) != shard.get("sha256"):
            raise IntegrityError(f"shard hash mismatch: {shard_path}")
        rows.extend(stream_jsonl(shard_path))
    return rows


def verify_trainable_manifest(
    manifest_path: Path,
    *,
    expected_policy_hash: Optional[str] = None,
    expected_decontamination_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail closed on stale provenance and altered shards before any trainer reads data."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"data manifest is unreadable: {exc}") from exc
    if not verify_hashed_artifact(manifest):
        raise IntegrityError("data manifest artifact hash is invalid")
    if manifest.get("status") != "trainable" or manifest.get("mode") != "real":
        raise IntegrityError("data manifest is not a real trainable artifact")
    if expected_policy_hash is not None and manifest.get("policy_hash") != expected_policy_hash:
        raise IntegrityError("data manifest policy hash is stale")
    corpus_path = path.parent / "decontamination.json"
    try:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"decontamination corpus is unreadable: {exc}") from exc
    if not verify_hashed_artifact(corpus):
        raise IntegrityError("decontamination corpus artifact hash is invalid")
    corpus_hash = corpus["artifact_hash"]
    expected = expected_decontamination_hash or corpus_hash
    if manifest.get("decontamination_hash") != expected or corpus_hash != expected:
        raise IntegrityError("data manifest has a stale decontamination hash")
    if expected_policy_hash is not None and corpus.get("policy_hash") != expected_policy_hash:
        raise IntegrityError("decontamination corpus policy hash is stale")
    for shard in manifest.get("shards", []):
        shard_path = path.parent / shard["path"]
        if file_sha256(shard_path) != shard.get("sha256"):
            raise IntegrityError(f"shard hash mismatch: {shard_path}")
    for schema, expected in manifest.get("split_statistics", {}).items():
        train_ids: set[str] = set()
        validation_ids: set[str] = set()
        for shard in manifest.get("shards", []):
            if shard.get("schema") != schema:
                continue
            target = validation_ids if shard.get("split") == "validation" else train_ids
            target.update(row["content_id"] for row in stream_jsonl(path.parent / shard["path"]))
        if train_ids & validation_ids:
            raise IntegrityError(f"{schema} train and validation shards overlap")
        if len(train_ids) != int(expected.get("train", 0)) or len(validation_ids) != int(
            expected.get("validation", 0)
        ):
            raise IntegrityError(f"{schema} split statistics do not match shard contents")
    return manifest


def verify_data_manifest(
    manifest_path: Path,
    *,
    expected_policy_hash: Optional[str] = None,
    expected_decontamination_hash: Optional[str] = None,
) -> Dict[str, Any]:
    return verify_trainable_manifest(
        manifest_path,
        expected_policy_hash=expected_policy_hash,
        expected_decontamination_hash=expected_decontamination_hash,
    )


def build_decontamination_corpus(
    records: Iterable[Mapping[str, Any]],
    out_path: Path,
    *,
    sources: Sequence[Mapping[str, Any]],
    policy_hash: str,
) -> Dict[str, Any]:
    """Persist only canonical strings and hashes required to reject evaluation overlap."""
    values: set[str] = set()
    for record in records:
        for key in ("prompt", "context", "expected", "answer", "document"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                values.add(" ".join(value.split()))
        options = record.get("options")
        if isinstance(options, list):
            values.update(" ".join(str(value).split()) for value in options)
    canonical = sorted(values)
    body = {
        "schema_version": DECONTAMINATION_SCHEMA_VERSION,
        "sources": [dict(source) for source in sources],
        "policy_hash": policy_hash,
        "entries": [{"canonical": value, "sha256": content_id(value)} for value in canonical],
    }
    document = {**body, "artifact_hash": sha256_bytes(canonical_json(body))}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def lm_eval_decontamination_records(tasks: Sequence[str]) -> Iterator[Dict[str, Any]]:
    """Yield documents, rendered prompts, answers, and choices from pinned lm-eval tasks."""
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise RuntimeError("lm-eval decontamination requires the eval extra") from exc
    manager = TaskManager()
    loaded = manager.load_task_or_group(list(tasks))
    for task_name in tasks:
        task = loaded[task_name]
        if isinstance(task, dict):
            task = next(iter(task.values()))
        docs = None
        split_errors = []
        for split_name, available, getter in (
            ("validation", task.has_validation_docs, task.validation_docs),
            ("test", task.has_test_docs, task.test_docs),
            ("train", task.has_training_docs, task.training_docs),
        ):
            if not available():
                continue
            try:
                docs = getter()
            except KeyError as exc:
                split_errors.append(f"{split_name}: {exc}")
                continue
            break
        if docs is None:
            detail = "; ".join(split_errors) or "no split advertised"
            raise RuntimeError(f"lm-eval task {task_name} has no readable document split: {detail}")
        for document in docs:
            record = {
                "document": json.dumps(document, ensure_ascii=False, sort_keys=True, default=str)
            }
            try:
                record["prompt"] = str(task.doc_to_text(document))
                record["answer"] = str(task.doc_to_target(document))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"could not canonicalize lm-eval task {task_name}: {exc}"
                ) from exc
            choices = document.get("choices") if isinstance(document, dict) else None
            if isinstance(choices, list):
                record["options"] = [str(choice) for choice in choices]
            yield record


def decontaminate(row: Mapping[str, Any], corpus: Mapping[str, Any]) -> bool:
    """Return True only when no canonical evaluation string occurs in the row."""
    haystack = " ".join(_row_text(row).split()).lower()

    def strings(value: Any) -> Iterator[str]:
        if isinstance(value, str):
            yield " ".join(value.split()).lower()
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    fields = set(strings(row))
    for entry in corpus.get("entries", []):
        value = " ".join(str(entry.get("canonical", "")).split()).lower()
        if not value:
            continue
        # Short answers and multiple-choice options are too generic for substring matching.
        # They still reject an exact field-sized copy. Longer benchmark text also rejects when
        # embedded in a larger prompt, which catches copied evaluation examples with wrappers.
        token_count = len(_TOKEN_RE.findall(value))
        substring_match = token_count >= 8 and len(value) >= 32 and value in haystack
        if substring_match or value in fields:
            return False
    return True


def deterministic_weighted_interleave(
    groups: Mapping[str, Iterable[Mapping[str, Any]]],
    weights: Mapping[str, float],
    *,
    seed: int,
    token_count: Callable[[Mapping[str, Any]], int],
    repeat: Optional[Mapping[str, bool]] = None,
    maximum_tokens: Optional[int] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Stream a reproducible weighted mix without silently repeating exhausted sources."""
    import random

    if set(groups) != set(weights) or any(weight < 0 for weight in weights.values()):
        raise ValueError("groups and non-negative weights must have identical keys")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one interleave weight must be positive")
    normalized = {group: weight / total for group, weight in weights.items()}
    rng = random.Random(seed)
    ordered = sorted(groups)
    iterators = {group: iter(groups[group]) for group in ordered}
    factories = {group: groups[group] for group in ordered}
    repeated: Counter[str] = Counter()
    actual_tokens: Counter[str] = Counter()
    yielded_tokens = 0
    active = set(ordered)
    while active and (maximum_tokens is None or yielded_tokens < maximum_tokens):
        available = [group for group in ordered if group in active and normalized[group] > 0]
        if not available:
            break
        pick = rng.random() * sum(normalized[group] for group in available)
        cumulative = 0.0
        selected = available[-1]
        for group in available:
            cumulative += normalized[group]
            if pick <= cumulative:
                selected = group
                break
        try:
            row = next(iterators[selected])
        except StopIteration:
            if (repeat or {}).get(selected, False):
                iterators[selected] = iter(factories[selected])
                repeated[selected] += 1
                try:
                    row = next(iterators[selected])
                except StopIteration:
                    active.remove(selected)
                    continue
            else:
                active.remove(selected)
                continue
        count = int(token_count(row))
        if count < 0:
            raise ValueError("token_count returned a negative value")
        actual_tokens[selected] += count
        yielded_tokens += count
        output = dict(row)
        output["_mix_metrics"] = {
            "source_group": selected,
            "group_tokens": actual_tokens[selected],
            "total_tokens": yielded_tokens,
            "repeat_count": repeated[selected],
        }
        yield selected, output


from .secure_compat.data_pipeline import (  # noqa: E402, F401
    DataError,
    LanguageIdentifier,
    classify_schema,
    deduplicate,
    discover,
    leakage_filter,
    load_license_reviews,
    normalize_license,
    normalize_text,
    prepare,
    reviewed_license,
    row_texts,
    run_cli as secure_run_cli,
)
