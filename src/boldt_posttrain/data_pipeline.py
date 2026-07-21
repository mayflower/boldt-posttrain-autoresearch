"""Revision-pinned Hugging Face discovery and fail-closed data materialization."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from . import config as config_module
from . import provenance
from .artifacts import (
    ArtifactRef,
    EventLog,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    exclusive_lock,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
    verify_artifact_ref,
)
from .evaluation import load_suite, suite_hash
from .policy import Policy, load_policy

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs/posttrain"
LICENSE_ALIASES = {
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "cc0-1.0": "CC0-1.0",
    "cc-by-4.0": "CC-BY-4.0",
}
ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
}


class DataError(RuntimeError):
    """A source or row is not safe and complete enough for training."""


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise DataError("text value must be a string")
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def normalize_license(value: Any) -> str | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return LICENSE_ALIASES.get(value.strip().casefold()) if isinstance(value, str) else None


def load_license_reviews(policy: Policy) -> tuple[dict[str, Any], Path]:
    path = ROOT / policy.document["data"]["license_reviews_path"]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"protected license review file is unreadable: {exc}") from exc
    if (
        set(document) != {"schema_version", "reviews"}
        or document["schema_version"] != 1
        or not isinstance(document["reviews"], dict)
    ):
        raise DataError("protected license review schema is invalid")
    return document["reviews"], path


def reviewed_license(
    dataset_id: str,
    revision: str,
    card_license: Any,
    policy: Policy,
    reviews: Mapping[str, Any],
) -> str | None:
    normalized = normalize_license(card_license)
    if normalized in policy.document["data"]["allowed_licenses"]:
        return normalized
    review = reviews.get(dataset_id)
    if not isinstance(review, dict):
        return None
    if set(review) != {"revision", "license", "approved", "reviewed_by"}:
        raise DataError(f"manual license review schema is invalid for {dataset_id}")
    reviewed = normalize_license(review["license"])
    if (
        review["approved"] is True
        and review["revision"] == revision
        and isinstance(review["reviewed_by"], str)
        and review["reviewed_by"].strip()
        and reviewed in policy.document["data"]["allowed_licenses"]
    ):
        return reviewed
    return None


def classify_schema(row: Mapping[str, Any]) -> str | None:
    keys = set(row)
    if "messages" in keys or "conversations" in keys or {"prompt", "response"} <= keys:
        return "sft"
    if {"prompt", "chosen", "rejected"} <= keys:
        return "preference"
    if len(keys & {"text", "document", "content"}) == 1:
        return "cpt"
    return None


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise DataError("conversation must be an array")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise DataError("conversation item must be an object")
        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if role not in ROLE_ALIASES or not isinstance(content, str):
            raise DataError("conversation role or content is invalid")
        normalized = normalize_text(content)
        if not normalized:
            raise DataError("conversation content is empty")
        messages.append({"role": ROLE_ALIASES[role], "content": normalized})
    roles = {item["role"] for item in messages}
    if not {"user", "assistant"} <= roles:
        raise DataError("SFT conversation requires user and assistant content")
    return messages


def normalize_row(row: Mapping[str, Any], source: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    schema = classify_schema(row)
    if schema is None:
        raise DataError("row schema is not an explicitly supported form")
    normalized: dict[str, Any] = {
        "type": schema,
        "source": {
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "config": source["config"],
            "split": source["split"],
            "row_id": str(row_id),
        },
        "license": source["license"],
    }
    if schema == "sft":
        if "messages" in row:
            normalized["messages"] = _messages(row["messages"])
        elif "conversations" in row:
            normalized["messages"] = _messages(row["conversations"])
        else:
            prompt, response = normalize_text(row["prompt"]), normalize_text(row["response"])
            if not prompt or not response:
                raise DataError("SFT prompt and response must be non-empty")
            normalized["messages"] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
    elif schema == "preference":
        prompt = normalize_text(row["prompt"])
        chosen = normalize_text(row["chosen"])
        rejected = normalize_text(row["rejected"])
        if not prompt or not chosen or not rejected or chosen == rejected:
            raise DataError("preference prompt/chosen/rejected must be non-empty and distinct")
        normalized.update(prompt=prompt, chosen=chosen, rejected=rejected)
    else:
        key = next(key for key in ("text", "document", "content") if key in row)
        text = normalize_text(row[key])
        if not text:
            raise DataError("CPT text must be non-empty")
        normalized["text"] = text
    content = {key: value for key, value in normalized.items() if key not in {"source", "license"}}
    normalized["content_id"] = sha256_bytes(canonical_json_bytes(content))
    return normalized


def row_texts(row: Mapping[str, Any]) -> list[str]:
    if row["type"] == "sft":
        return [item["content"] for item in row["messages"]]
    if row["type"] == "preference":
        return [row["prompt"], row["chosen"], row["rejected"]]
    return [row["text"]]


class LanguageIdentifier:
    def __init__(self, policy: Policy):
        import fasttext

        descriptor = policy.document["data"]["language_model"]
        self.path = ROOT / descriptor["path"]
        if sha256_file(self.path) != descriptor["sha256"]:
            raise DataError("language model hash mismatch")
        self.model = fasttext.load_model(str(self.path))
        self.minimum = float(policy.document["data"]["min_german_confidence"])

    def check(self, text: str) -> tuple[bool, float]:
        cleaned = re.sub(r"https?://\S+|```.*?```", " ", text, flags=re.DOTALL).strip()
        if len(cleaned) < 20:
            return False, 0.0
        labels, probabilities = self.model.predict(cleaned.replace("\n", " "), k=1)
        confidence = float(probabilities[0])
        return labels[0] == "__label__de" and confidence >= self.minimum, confidence


def _shingles(text: str, size: int = 5) -> set[str]:
    words = re.findall(r"\w+", normalize_text(text).casefold())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _minhash(text: str):
    from datasketch import MinHash

    value = MinHash(num_perm=128, seed=1729)
    for shingle in sorted(_shingles(text)):
        value.update(shingle.encode())
    return value


def deduplicate(
    rows: Iterable[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasketch import MinHashLSH

    exact: set[str] = set()
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[dict[str, Any]] = []
    stats = {"exact_removed": 0, "near_removed": 0}
    for row in rows:
        if row["content_id"] in exact:
            stats["exact_removed"] += 1
            continue
        signature = _minhash("\n".join(row_texts(row)))
        if lsh.query(signature):
            stats["near_removed"] += 1
            continue
        lsh.insert(f"row-{len(kept)}", signature)
        exact.add(row["content_id"])
        kept.append(row)
    return kept, stats


def leakage_filter(
    rows: Iterable[dict[str, Any]], policy: Policy
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasketch import MinHashLSH

    cases = load_suite()
    exact = {
        sha256_bytes(normalize_text(case["prompt"]).casefold().encode()): case["case_id"]
        for case in cases
    }
    lsh = MinHashLSH(threshold=float(policy.document["data"]["leakage_jaccard"]), num_perm=128)
    for case in cases:
        lsh.insert(case["case_id"], _minhash(case["prompt"]))
    clean: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for row in rows:
        row_hits: list[dict[str, str]] = []
        for text in row_texts(row):
            normalized = normalize_text(text).casefold()
            text_hash = sha256_bytes(normalized.encode())
            exact_id = exact.get(text_hash)
            if exact_id:
                row_hits.append({"type": "exact", "case_id": exact_id, "text_sha256": text_hash})
            for case_id in sorted(lsh.query(_minhash(text))):
                if case_id != exact_id:
                    row_hits.append({"type": "near", "case_id": case_id, "text_sha256": text_hash})
        if row_hits:
            hits.append({"content_id": row["content_id"], "matches": row_hits})
        else:
            clean.append(row)
    report = {
        "status": "clean" if not hits else "leak_detected",
        "hits": hits,
        "hit_count": len(hits),
        "suite_hash": suite_hash(),
    }
    return clean, report


def _card_license(info: Any) -> Any:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return card.get("license") if hasattr(card, "get") else getattr(card, "license", None)


def _row_estimate(info: Any, config: str, split: str) -> int | None:
    card_data = getattr(info, "card_data", None)
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    if not isinstance(card_data, dict):
        return None
    descriptors = card_data.get("dataset_info", [])
    if isinstance(descriptors, dict):
        descriptors = [descriptors]
    for descriptor in descriptors if isinstance(descriptors, list) else []:
        if not isinstance(descriptor, dict):
            continue
        descriptor_config = descriptor.get("config_name", descriptor.get("name", "default"))
        if descriptor_config != config and not (config == "default" and descriptor_config is None):
            continue
        splits = descriptor.get("splits", {})
        if isinstance(splits, list):
            splits = {
                item.get("name"): item
                for item in splits
                if isinstance(item, dict) and item.get("name")
            }
        details = splits.get(split) if isinstance(splits, dict) else None
        count = details.get("num_examples") if isinstance(details, dict) else None
        if isinstance(count, int) and count >= 0:
            return count
    return None


def discover(policy: Policy, *, api=None, sample_limit: int = 20) -> dict[str, Any]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    api = api or HfApi()
    language = LanguageIdentifier(policy)
    reviews, review_path = load_license_reviews(policy)
    candidates: list[dict[str, Any]] = []
    for organization in policy.document["data"]["allowed_organizations"]:
        for listed in api.list_datasets(author=organization, full=True):
            revision = getattr(listed, "sha", None)
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise DataError(f"dataset has no immutable revision: {listed.id}")
            info = api.dataset_info(listed.id, revision=revision, files_metadata=True)
            if getattr(info, "sha", None) != revision:
                raise DataError(f"Hub resolved a different dataset revision: {listed.id}")
            script_files = [
                item.rfilename
                for item in getattr(info, "siblings", [])
                if item.rfilename.endswith(".py")
            ]
            card_license = _card_license(info)
            spdx = reviewed_license(listed.id, revision, card_license, policy, reviews)
            try:
                configs = get_dataset_config_names(listed.id, revision=revision)
            except Exception as exc:
                candidates.append(
                    {
                        "dataset_id": listed.id,
                        "dataset_revision_sha": revision,
                        "training_usable": False,
                        "rejection_reasons": [f"config_resolution_failed:{type(exc).__name__}"],
                    }
                )
                continue
            for config in configs:
                for split in get_dataset_split_names(
                    listed.id, config_name=config, revision=revision
                ):
                    reasons: list[str] = []
                    if (
                        script_files
                        and listed.id not in policy.document["data"]["remote_code_allowlist"]
                    ):
                        reasons.append("remote_code_required")
                    if spdx not in policy.document["data"]["allowed_licenses"]:
                        reasons.append("license_not_allowed")
                    if bool(getattr(info, "gated", False)) or bool(getattr(info, "private", False)):
                        reasons.append("gated_or_private")
                    sample: list[dict[str, Any]] = []
                    if "remote_code_required" not in reasons:
                        stream = load_dataset(
                            listed.id,
                            name=config,
                            split=split,
                            revision=revision,
                            streaming=True,
                        )
                        try:
                            for index, row in enumerate(stream):
                                if index >= sample_limit:
                                    break
                                sample.append(dict(row))
                        except Exception as exc:
                            reasons.append(f"sample_stream_failed:{type(exc).__name__}")
                    schemas = Counter(classify_schema(row) or "unknown" for row in sample)
                    checks = [
                        language.check(
                            "\n".join(
                                str(value) for value in row.values() if isinstance(value, str)
                            )
                        )
                        for row in sample
                    ]
                    candidates.append(
                        {
                            "dataset_id": listed.id,
                            "dataset_revision_sha": revision,
                            "config": config,
                            "split": split,
                            "row_estimate": _row_estimate(info, config, split),
                            "dataset_card_license": card_license,
                            "normalized_spdx_license": spdx,
                            "gated": bool(getattr(info, "gated", False)),
                            "private": bool(getattr(info, "private", False)),
                            "schema_fingerprint": sha256_bytes(
                                canonical_json_bytes(
                                    sorted(
                                        (key, type(value).__name__)
                                        for row in sample
                                        for key, value in row.items()
                                    )
                                )
                            ),
                            "schema_classification": schemas.most_common(1)[0][0]
                            if schemas
                            else "unknown",
                            "language_evidence": {
                                "checked": len(checks),
                                "german": sum(ok for ok, _ in checks),
                                "mean_confidence": sum(score for _, score in checks) / len(checks)
                                if checks
                                else 0.0,
                            },
                            "sample_hash": sha256_bytes(canonical_json_bytes(sample)),
                            "training_usable": not reasons and bool(sample),
                            "rejection_reasons": reasons,
                        }
                    )
    return {
        "schema_version": 1,
        "license_reviews_sha256": sha256_file(review_path),
        "candidates": candidates,
    }


def _source_rows(source: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset(
        source["dataset_id"],
        name=source["config"],
        split=source["split"],
        revision=source["revision"],
        streaming=True,
    )
    for row in stream:
        yield dict(row)


def _published_ref(source: Path, destination: Path, role: str) -> ArtifactRef:
    media_type = "application/jsonl" if source.suffix == ".jsonl" else "application/json"
    measured = ArtifactRef.from_path(source, role=role, media_type=media_type)
    try:
        stored_path = destination.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        stored_path = str(destination.resolve())
    return ArtifactRef(
        stored_path,
        measured.kind,
        measured.role,
        measured.sha256,
        measured.size_bytes,
        measured.media_type,
    )


def verify_data_manifest(
    data_root: Path, policy: Policy, *, repository_root: Path = ROOT
) -> dict[str, Any]:
    pointer_path = data_root / "current.json"
    if not pointer_path.is_file():
        raise DataError("data current pointer is missing")
    pointer = json.loads(pointer_path.read_text())
    if (
        set(pointer)
        != {
            "schema_version",
            "run_id",
            "manifest_sha256",
            "run_card_sha256",
        }
        or pointer.get("schema_version") != 1
    ):
        raise DataError("data current pointer schema is invalid")
    run_dir = data_root / pointer["run_id"]
    manifest_path, run_card_path = run_dir / "manifest.json", run_dir / "run_card.json"
    if sha256_file(manifest_path) != pointer["manifest_sha256"]:
        raise DataError("data manifest hash mismatch")
    if sha256_file(run_card_path) != pointer["run_card_sha256"]:
        raise DataError("data run-card hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "trainable":
        raise DataError("data manifest is not trainable")
    if manifest.get("eval_suite_hash") != suite_hash():
        raise DataError("data manifest eval-suite fingerprint is missing or stale")
    if manifest.get("policy_sha256") != sha256_file(policy.path):
        raise DataError("data manifest policy hash mismatch")
    if manifest.get("leakage_statistics") != {"status": "clean", "hit_count": 0}:
        raise DataError("data manifest leakage gate is not clean")
    refs = [*manifest.get("shards", []), *manifest.get("reports", [])]
    if not refs:
        raise DataError("data manifest contains no artifact references")
    for ref in refs:
        verify_artifact_ref(ref, root=repository_root)
    return manifest


def prepare(policy: Policy, config_path: Path, *, rows_provider=_source_rows) -> dict[str, Any]:
    from huggingface_hub import HfApi

    config = config_module.load_experiment(config_path)
    sources = config.document["data"]["sources"]
    if not sources:
        raise DataError("experiment data.sources is empty")
    maximum = min(config.document["data"]["max_rows"], policy.document["data"]["max_total_rows"])
    reviews, license_reviews = load_license_reviews(policy)
    language = LanguageIdentifier(policy)
    started = time.monotonic()
    run_id = new_run_id("data-prepare")
    output_root = OUTPUTS / "data"
    staging, final = output_root / ".staging" / run_id, output_root / run_id
    staging.mkdir(parents=True)
    events = EventLog(OUTPUTS)
    start = events.append("run_started", run_id, {"run_type": "data_prepare"})
    normalized: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    confidences: list[float] = []
    discovery_entries: list[dict[str, Any]] = []
    before = 0
    try:
        for source in sources:
            allowed_prefixes = tuple(
                f"{org}/" for org in policy.document["data"]["allowed_organizations"]
            )
            if not source["dataset_id"].startswith(allowed_prefixes):
                raise DataError(f"source organization is not allowed: {source['dataset_id']}")
            if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
                raise DataError(f"source revision is not exact: {source['dataset_id']}")
            info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"])
            if getattr(info, "sha", None) != source["revision"]:
                raise DataError(f"Hub resolved a different revision: {source['dataset_id']}")
            if bool(getattr(info, "gated", False)) or bool(getattr(info, "private", False)):
                raise DataError(f"source is gated or private: {source['dataset_id']}")
            script_files = [
                item.rfilename
                for item in getattr(info, "siblings", [])
                if item.rfilename.endswith(".py")
            ]
            if (
                script_files
                and source["dataset_id"] not in policy.document["data"]["remote_code_allowlist"]
            ):
                raise DataError(f"source requires forbidden remote code: {source['dataset_id']}")
            card_license = _card_license(info)
            license_value = reviewed_license(
                source["dataset_id"],
                source["revision"],
                card_license,
                policy,
                reviews,
            )
            if license_value not in policy.document["data"]["allowed_licenses"]:
                raise DataError(f"source license is unknown or forbidden: {source['dataset_id']}")
            discovery_entries.append(
                {
                    "dataset_id": source["dataset_id"],
                    "dataset_revision_sha": source["revision"],
                    "config": source["config"],
                    "split": source["split"],
                    "dataset_card_license": card_license,
                    "normalized_spdx_license": license_value,
                }
            )
            descriptor = {**source, "license": license_value}
            source_rows = 0
            try:
                for row_index, raw in enumerate(rows_provider(source)):
                    if (
                        before >= maximum
                        or source_rows >= policy.document["data"]["max_rows_per_source"]
                    ):
                        break
                    before += 1
                    source_rows += 1
                    try:
                        item = normalize_row(raw, descriptor, str(row_index))
                        if item["type"] != source["schema"]:
                            raise DataError("row_schema_mismatch")
                        ok, confidence = language.check("\n".join(row_texts(item)))
                        confidences.append(confidence)
                        if not ok:
                            raise DataError("language_not_german")
                        normalized.append(item)
                    except DataError as exc:
                        rejections[str(exc)] += 1
            except Exception as exc:
                raise DataError(
                    f"source stream interrupted: {source['dataset_id']}: {exc}"
                ) from exc
            if before >= maximum:
                break
        deduplicated, dedup_stats = deduplicate(
            normalized, policy.document["data"]["near_dedup_jaccard"]
        )
        clean, leakage = leakage_filter(deduplicated, policy)
        if leakage["status"] != "clean":
            raise DataError("benchmark leakage detected")
        if not clean:
            raise DataError("no trainable rows remain after policy filters")
        groups = {
            kind: [row for row in clean if row["type"] == kind]
            for kind in ("sft", "preference", "cpt")
        }
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            policy.seed_model["repo_id"],
            revision=policy.seed_model["revision"],
            local_files_only=Path(policy.seed_model["repo_id"]).is_absolute(),
        )
        token_lengths = {
            row["content_id"]: len(
                tokenizer(
                    "\n".join(row_texts(row)),
                    add_special_tokens=False,
                )["input_ids"]
            )
            for row in clean
        }
        ordered_lengths = sorted(token_lengths.values())
        cpt_tokens = sum(token_lengths[row["content_id"]] for row in groups["cpt"])
        total_tokens = sum(ordered_lengths)
        if cpt_tokens > policy.document["data"]["max_cpt_tokens"]:
            raise DataError("CPT token count exceeds protected maximum")
        if total_tokens and cpt_tokens / total_tokens > policy.document["data"]["max_cpt_fraction"]:
            raise DataError("CPT token fraction exceeds protected maximum")
        names = {
            "sft": "train_sft-00000-of-00001.jsonl",
            "preference": "train_preference-00000-of-00001.jsonl",
            "cpt": "train_cpt-00000-of-00001.jsonl",
        }
        shard_refs: list[ArtifactRef] = []
        for kind, rows in groups.items():
            path = staging / names[kind]
            atomic_write_bytes(path, b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
            shard_refs.append(_published_ref(path, final / path.name, f"{kind}_shard"))
        quality = {
            "schema_version": 1,
            "rows_before": before,
            "rows_normalized": len(normalized),
            "rows_trainable": len(clean),
            "rejections": dict(rejections),
            "language": {
                "checked": len(confidences),
                "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            },
            "dedup": dedup_stats,
            "types": {kind: len(rows) for kind, rows in groups.items()},
            "token_lengths": {
                "count": len(ordered_lengths),
                "total": total_tokens,
                "minimum": min(ordered_lengths) if ordered_lengths else 0,
                "maximum": max(ordered_lengths) if ordered_lengths else 0,
                "mean": total_tokens / len(ordered_lengths) if ordered_lengths else 0.0,
                "cpt_tokens": cpt_tokens,
            },
        }
        atomic_write_json(
            staging / "discovery.json",
            {"schema_version": 1, "sources": discovery_entries},
        )
        atomic_write_json(staging / "leakage_report.json", leakage)
        atomic_write_json(staging / "quality_report.json", quality)
        leakage_ref = _published_ref(
            staging / "leakage_report.json", final / "leakage_report.json", "leakage_report"
        )
        quality_ref = _published_ref(
            staging / "quality_report.json", final / "quality_report.json", "quality_report"
        )
        discovery_ref = _published_ref(
            staging / "discovery.json", final / "discovery.json", "data_discovery"
        )
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "trainable",
            "sources": sources,
            "shards": [ref.to_dict() for ref in shard_refs],
            "reports": [
                discovery_ref.to_dict(),
                leakage_ref.to_dict(),
                quality_ref.to_dict(),
            ],
            "license_status": "usable",
            "language_statistics": quality["language"],
            "dedup_statistics": dedup_stats,
            "token_statistics": quality["token_lengths"],
            "leakage_statistics": {
                "status": leakage["status"],
                "hit_count": leakage["hit_count"],
            },
            "eval_suite_hash": suite_hash(),
            "policy_sha256": sha256_file(policy.path),
            "config_sha256": sha256_file(config.path),
            "license_reviews_sha256": sha256_file(license_reviews),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        manifest_ref = _published_ref(
            staging / "manifest.json", final / "manifest.json", "data_manifest"
        )
        outputs = [*shard_refs, discovery_ref, leakage_ref, quality_ref, manifest_ref]
        card = {
            "schema_version": 1,
            "run_id": run_id,
            "run_type": "data_prepare",
            "mode": "real",
            "status": "succeeded",
            "started_at": start["event"]["timestamp"],
            "finished_at": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "duration_seconds": time.monotonic() - started,
            "command": ["python", "-m", "boldt_posttrain.cli", "data", "prepare", "--real"],
            "git": provenance.collect_git("HEAD"),
            "policy": {"path": str(policy.path), "sha256": manifest["policy_sha256"]},
            "experiment": {
                "path": str(config.path),
                "sha256": manifest["config_sha256"],
                "resolved_sha256": sha256_bytes(canonical_json_bytes(config.document)),
            },
            "inputs": [],
            "outputs": [ref.to_dict() for ref in outputs],
            "model": {},
            "data": manifest,
            "parameters": {"max_rows": maximum},
            "hardware": provenance.collect_hardware(),
            "environment": {
                **provenance.collect_environment(),
                "event_head": {
                    key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")
                },
            },
            "parents": [],
            "compatibility_fingerprint": sha256_bytes(
                canonical_json_bytes({"suite": manifest["eval_suite_hash"], "sources": sources})
            ),
            "error": None,
        }
        validate_run_card(card)
        atomic_write_json(staging / "run_card.json", card)
        os.replace(staging, final)
        pointer = {
            "schema_version": 1,
            "run_id": run_id,
            "manifest_sha256": sha256_file(final / "manifest.json"),
            "run_card_sha256": sha256_file(final / "run_card.json"),
        }
        with exclusive_lock(output_root / ".data.lock"):
            atomic_write_json(output_root / "current.json", pointer)
        finish = events.append(
            "run_finished",
            run_id,
            {
                "status": "succeeded",
                "run_card_sha256": sha256_file(final / "run_card.json"),
            },
        )
        return {
            "status": "succeeded",
            "run_id": run_id,
            "manifest": str(final / "manifest.json"),
            "event_sequence": finish["sequence"],
        }
    except Exception:
        events.append("run_finished", run_id, {"status": "failed"})
        raise


def run_cli(args) -> tuple[dict[str, Any], int]:
    policy = load_policy()
    if args.data_command == "discover":
        run_id = new_run_id("data-discover")
        staging = OUTPUTS / "data/.staging" / run_id
        final = OUTPUTS / "data" / run_id
        staging.mkdir(parents=True)
        events = EventLog(OUTPUTS)
        start = events.append("run_started", run_id, {"run_type": "data_discover"})
        document = discover(policy)
        atomic_write_json(staging / "discovery.json", document)
        discovery_ref = _published_ref(
            staging / "discovery.json", final / "discovery.json", "data_discovery"
        )
        config = config_module.load_experiment(ROOT / args.config)
        card = {
            "schema_version": 1,
            "run_id": run_id,
            "run_type": "data_discover",
            "mode": "real",
            "status": "succeeded",
            "started_at": start["event"]["timestamp"],
            "finished_at": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "duration_seconds": 0.0,
            "command": ["python", "-m", "boldt_posttrain.cli", "data", "discover", "--real"],
            "git": provenance.collect_git("HEAD"),
            "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
            "experiment": {
                "path": str(config.path),
                "sha256": sha256_file(config.path),
                "resolved_sha256": sha256_bytes(canonical_json_bytes(config.document)),
            },
            "inputs": [],
            "outputs": [discovery_ref.to_dict()],
            "model": {},
            "data": document,
            "parameters": {"sample_limit": 20},
            "hardware": provenance.collect_hardware(),
            "environment": {
                **provenance.collect_environment(),
                "event_head": {
                    key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")
                },
            },
            "parents": [],
            "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(document)),
            "error": None,
        }
        validate_run_card(card)
        atomic_write_json(staging / "run_card.json", card)
        os.replace(staging, final)
        finish = events.append(
            "run_finished",
            run_id,
            {"status": "succeeded", "run_card_sha256": sha256_file(final / "run_card.json")},
        )
        return {
            "status": "succeeded",
            "run_id": run_id,
            "discovery": str(final / "discovery.json"),
            "event_sequence": finish["sequence"],
        }, 0
    return prepare(policy, ROOT / args.config), 0
