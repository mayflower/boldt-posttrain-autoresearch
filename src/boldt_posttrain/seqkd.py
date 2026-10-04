"""Sequence-level distillation (seqkd) from an external, policy-approved teacher.

Logit distillation (`opd`/`distill`) needs a teacher that shares the student's
tokenizer, and no larger model shares Boldt's. Sequence-level distillation crosses
the boundary as text instead:

1. ``pt seqkd generate`` lets the teacher answer prompts drawn from the verified SFT
   manifest and publishes the answers as a trainable SFT manifest, after the same
   language, dedup and benchmark-leakage gates that ``data prepare`` applies.
2. The ``seqkd`` lever trains the student with plain SFT on exactly one pinned,
   event-anchored generation run.

The teacher is never chosen by the experiment file alone: it must be an exact
``repo_id@commit`` entry in ``policy.teachers``. Generation runs outside the loop
budget because a 27B teacher needs longer than one 90-minute round.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import (
    RUN_ID_RE,
    ArtifactError,
    ArtifactRef,
    EventLog,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
    verify_artifact_ref,
)
from .policy import Policy
from .resolver import HUB_REF_RE, OUTPUTS, ROOT, _verified_event_runs

THINK_MARKERS = ("<think>", "</think>")
SHARD_NAME = "train_sft-00000-of-00001.jsonl"

_SETTINGS_SCHEMA: dict[str, tuple[type, ...]] = {
    "teacher": (str,),
    "generation_run": (str,),
    "max_prompts": (int,),
    "max_new_tokens": (int,),
    "temperature": (int, float),
    "top_p": (int, float),
    "top_k": (int,),
    "presence_penalty": (int, float),
    "seed": (int,),
}

# One generated answer per prompt: text plus the engine's stop reason and lengths.
Generation = Mapping[str, Any]
Generator = Callable[..., list[Generation]]


class SeqKDError(RuntimeError):
    """The teacher, its generations, or a pinned generation run violates the contract."""


def seqkd_settings(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the experiment's ``seqkd`` block exactly; no defaults, no extra keys."""
    settings = document.get("seqkd")
    if not isinstance(settings, dict):
        raise ValueError("config.seqkd must be an object for the seqkd lever")
    unknown = sorted(set(settings) - set(_SETTINGS_SCHEMA))
    missing = sorted(set(_SETTINGS_SCHEMA) - set(settings))
    if unknown or missing:
        raise ValueError(f"config.seqkd keys invalid; missing={missing} unknown={unknown}")
    for key, types in _SETTINGS_SCHEMA.items():
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, types):
            raise ValueError(f"config.seqkd.{key} has an invalid type")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"config.seqkd.{key} must be finite")
    if not HUB_REF_RE.fullmatch(settings["teacher"]):
        raise ValueError("config.seqkd.teacher must be repo_id@40-character-commit")
    if settings["generation_run"] and not RUN_ID_RE.fullmatch(settings["generation_run"]):
        raise ValueError("config.seqkd.generation_run must be empty or an exact run ID")
    if settings["max_prompts"] < 1 or settings["max_new_tokens"] < 1:
        raise ValueError("config.seqkd.max_prompts and max_new_tokens must be positive")
    if not 0 <= settings["temperature"] <= 2:
        raise ValueError("config.seqkd.temperature must satisfy 0 <= value <= 2")
    if not 0 < settings["top_p"] <= 1:
        raise ValueError("config.seqkd.top_p must satisfy 0 < value <= 1")
    if settings["top_k"] != -1 and settings["top_k"] < 1:
        raise ValueError("config.seqkd.top_k must be -1 (disabled) or positive")
    if not -2 <= settings["presence_penalty"] <= 2:
        raise ValueError("config.seqkd.presence_penalty must satisfy -2 <= value <= 2")
    return dict(settings)


def generation_parameters(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings that determine a generation run (everything but the run pin)."""
    return {key: value for key, value in settings.items() if key != "generation_run"}


def validate_seqkd_policy(document: Mapping[str, Any], policy: Policy) -> dict[str, Any]:
    allowed = policy.document["training"]["allowed_methods"]
    if "seqkd" not in allowed:
        raise ValueError("seqkd is not authorized by policy.training.allowed_methods")
    training = document["training"]
    if training["method"] not in {"lora", "qlora"} or training["method"] not in allowed:
        raise ValueError(
            "seqkd training requires an explicitly policy-allowed lora or qlora method"
        )
    if training["specialist"] not in policy.document["training"]["allowed_specialists"]:
        raise ValueError("seqkd specialist is not policy-allowed")
    settings = seqkd_settings(document)
    policy_teacher(policy, settings["teacher"])
    return settings


def policy_teacher(policy: Policy, reference: str) -> dict[str, Any]:
    match = HUB_REF_RE.fullmatch(reference)
    if not match:
        raise SeqKDError("seqkd teacher must be repo_id@40-character-commit")
    for entry in policy.document["teachers"]:
        if (
            entry["purpose"] == "seqkd"
            and entry["repo_id"] == match.group("repo")
            and entry["revision"] == match.group("revision")
        ):
            return dict(entry)
    raise SeqKDError(f"seqkd teacher {reference} is not an approved entry in policy.teachers")


def _shard_rows(manifest: Mapping[str, Any], role: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in manifest["shards"]:
        if ref["role"] != role:
            continue
        verify_artifact_ref(ref, root=root)
        path = Path(ref["path"])
        path = path if path.is_absolute() else root / path
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    return rows


def prompt_prefix(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]] | None:
    """The conversation up to the first assistant turn, which the teacher re-answers."""
    prefix: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "assistant":
            break
        prefix.append({"role": message["role"], "content": message["content"]})
    if not prefix or prefix[-1]["role"] != "user":
        return None
    return prefix


def select_prompts(
    rows: Sequence[Mapping[str, Any]], *, maximum: int, seed: int
) -> list[tuple[Mapping[str, Any], list[dict[str, str]]]]:
    """Deterministic, order-independent sample of answerable SFT prompts."""
    candidates = []
    for row in sorted(rows, key=lambda item: item["content_id"]):
        prefix = prompt_prefix(row["messages"])
        if prefix is not None:
            candidates.append((row, prefix))
    if len(candidates) > maximum:
        indices = sorted(random.Random(seed).sample(range(len(candidates)), maximum))
        candidates = [candidates[index] for index in indices]
    return candidates


def rejection_reason(generation: Generation) -> str | None:
    text = generation.get("text")
    if generation.get("finish_reason") == "prompt_too_long":
        return "prompt_too_long"
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if generation.get("finish_reason") != "stop":
        return "truncated"
    if any(marker in text for marker in THINK_MARKERS):
        return "thinking_leak"
    return None


def build_rows(
    selected: Sequence[tuple[Mapping[str, Any], list[dict[str, str]]]],
    generations: Sequence[Generation],
    *,
    language,
    teacher: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str], list[float]]:
    from .data_pipeline import DataError, normalize_row, row_texts

    if len(generations) != len(selected):
        raise SeqKDError("teacher returned a different number of answers than prompts")
    rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    confidences: list[float] = []
    for (source_row, prefix), generation in zip(selected, generations):
        reason = rejection_reason(generation)
        if reason is not None:
            rejections[reason] += 1
            continue
        source = source_row["source"]
        descriptor = {
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "config": source["config"],
            "split": source["split"],
            "license": source_row["license"],
        }
        raw = {"messages": [*prefix, {"role": "assistant", "content": generation["text"]}]}
        try:
            item = normalize_row(raw, descriptor, source["row_id"])
        except DataError as exc:
            rejections[str(exc)] += 1
            continue
        ok, confidence = language.check("\n".join(row_texts(item)))
        confidences.append(confidence)
        if not ok:
            rejections["language_not_german"] += 1
            continue
        item["teacher"] = {"repo_id": teacher["repo_id"], "revision": teacher["revision"]}
        item["prompt_content_id"] = source_row["content_id"]
        rows.append(item)
    return rows, rejections, confidences


def vllm_generator(
    prompts: Sequence[list[dict[str, str]]],
    *,
    settings: Mapping[str, Any],
    teacher: Mapping[str, Any],
    max_model_len: int,
) -> list[dict[str, Any]]:
    """Offline vLLM generation; thinking is disabled through the chat template."""
    # FlashInfer's top-k/top-p sampler JIT-compiles CUDA kernels and needs nvcc; the
    # training hosts ship only the driver. vLLM's PyTorch sampler is exact, not a
    # different algorithm. setdefault keeps an operator override.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=teacher["repo_id"],
        revision=teacher["revision"],
        tokenizer_revision=teacher["revision"],
        max_model_len=max_model_len,
        # Text-only distillation: skip the vision tower so its memory goes to KV cache.
        language_model_only=True,
        gpu_memory_utilization=0.90,
        # Qwen3.8's linear-attention layers need one Mamba cache block per running
        # sequence; on a 48 GB card about 220 fit, below vLLM's default of 256.
        max_num_seqs=128,
        seed=settings["seed"],
    )
    tokenizer = llm.get_tokenizer()
    budget = max_model_len - settings["max_new_tokens"]
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    runnable: list[int] = []
    for index, prompt in enumerate(prompts):
        # Render, then count: tokenize=True returns a dict.
        rendered = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if length > budget:
            results[index] = {
                "text": "",
                "finish_reason": "prompt_too_long",
                "prompt_tokens": length,
                "completion_tokens": 0,
            }
        else:
            runnable.append(index)
    params = SamplingParams(
        n=1,
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        top_k=settings["top_k"],
        presence_penalty=settings["presence_penalty"],
        max_tokens=settings["max_new_tokens"],
        seed=settings["seed"],
    )
    outputs = llm.chat(
        [prompts[index] for index in runnable],
        params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    for index, output in zip(runnable, outputs):
        completion = output.outputs[0]
        results[index] = {
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "prompt_tokens": len(output.prompt_token_ids or []),
            "completion_tokens": len(completion.token_ids),
        }
    return [item for item in results if item is not None]


def _published(source: Path, destination: Path, role: str, root: Path) -> ArtifactRef:
    media_type = "application/jsonl" if source.suffix == ".jsonl" else "application/json"
    measured = ArtifactRef.from_path(source, role=role, media_type=media_type)
    try:
        stored = destination.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        stored = str(destination.resolve())
    return ArtifactRef(
        stored, measured.kind, measured.role, measured.sha256, measured.size_bytes, media_type
    )


def _library_versions() -> dict[str, str | None]:
    import importlib.metadata

    versions: dict[str, str | None] = {}
    for name in ("vllm", "transformers", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def generate(
    policy: Policy,
    config_path: Path,
    *,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
    generator: Generator = vllm_generator,
    resolve_teacher: Callable[[str, Policy], Any] | None = None,
    teacher_license: Callable[..., str] | None = None,
    language=None,
) -> dict[str, Any]:
    """Publish one teacher-generated SFT manifest plus an event-anchored run card."""
    from .distillation import _teacher_license
    from .resolver import resolve_hub_model
    from . import config as config_module
    from . import provenance
    from .data_pipeline import (
        LanguageIdentifier,
        deduplicate,
        leakage_filter,
        verify_data_manifest,
    )
    from .evaluation import suite_hash

    config = config_module.load_experiment(config_path)
    settings = seqkd_settings(config.document)
    entry = policy_teacher(policy, settings["teacher"])
    resolved = (resolve_teacher or resolve_hub_model)(settings["teacher"], policy)
    license_id = (teacher_license or _teacher_license)(resolved, policy, None)
    if license_id != entry["license"]:
        raise SeqKDError("teacher model-card license differs from the policy-approved license")
    prompt_manifest = verify_data_manifest(
        outputs_root / "data", policy, repository_root=repository_root
    )
    source_rows = _shard_rows(prompt_manifest, "sft_shard", repository_root)
    selected = select_prompts(source_rows, maximum=settings["max_prompts"], seed=settings["seed"])
    if not selected:
        raise SeqKDError("the verified SFT manifest contains no answerable prompts")
    language = language or LanguageIdentifier(policy)
    max_model_len = int(config.document["training"]["context_length"])

    started = time.monotonic()
    run_id = new_run_id("seqkd-generate")
    output_root = outputs_root / "seqkd"
    staging, final = output_root / ".staging" / run_id, output_root / run_id
    run_staging, run_final = outputs_root / "runs/.staging" / run_id, outputs_root / "runs" / run_id
    events = EventLog(outputs_root)
    start = events.append("run_started", run_id, {"run_type": "seqkd_generate"})
    try:
        generation_started = time.monotonic()
        generations = generator(
            [prefix for _, prefix in selected],
            settings=settings,
            teacher=entry,
            max_model_len=max_model_len,
        )
        generation_seconds = time.monotonic() - generation_started
        rows, rejections, confidences = build_rows(
            selected, generations, language=language, teacher=entry
        )
        rows, dedup_stats = deduplicate(rows, policy.document["data"]["near_dedup_jaccard"])
        clean, leakage = leakage_filter(rows, policy)
        if leakage["status"] != "clean":
            raise SeqKDError("benchmark leakage detected in teacher answers")
        if not clean:
            raise SeqKDError("no teacher answer survived the policy filters")
        completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in generations)
        staging.mkdir(parents=True)
        shard = staging / SHARD_NAME
        atomic_write_bytes(shard, b"".join(canonical_json_bytes(row) + b"\n" for row in clean))
        report = {
            "schema_version": 1,
            "prompts_selected": len(selected),
            "answers_returned": len(generations),
            "rows_trainable": len(clean),
            "rejections": dict(sorted(rejections.items())),
            "finish_reasons": dict(
                sorted(Counter(str(item.get("finish_reason")) for item in generations).items())
            ),
            "language": {
                "checked": len(confidences),
                "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            },
            "dedup": dedup_stats,
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in generations),
            "completion_tokens": completion_tokens,
            "generation_seconds": generation_seconds,
            "completion_tokens_per_second": (
                completion_tokens / generation_seconds if generation_seconds > 0 else 0.0
            ),
            "libraries": _library_versions(),
        }
        atomic_write_json(staging / "generation_report.json", report)
        atomic_write_json(staging / "leakage_report.json", leakage)
        shard_ref = _published(shard, final / SHARD_NAME, "sft_shard", repository_root)
        report_ref = _published(
            staging / "generation_report.json",
            final / "generation_report.json",
            "seqkd_generation_report",
            repository_root,
        )
        leakage_ref = _published(
            staging / "leakage_report.json",
            final / "leakage_report.json",
            "leakage_report",
            repository_root,
        )
        teacher_record = {
            **entry,
            "tokenizer_sha256": resolved.tokenizer_sha256,
            "chat_template_sha256": resolved.chat_template_sha256,
            "model_config_sha256": resolved.model_config_sha256,
            "architecture": resolved.architecture,
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "trainable",
            "kind": "seqkd",
            "teacher": teacher_record,
            "prompt_source": {
                "run_id": prompt_manifest["run_id"],
                "manifest_sha256": sha256_bytes(canonical_json_bytes(prompt_manifest)),
            },
            "generation": generation_parameters(settings),
            "max_model_len": max_model_len,
            # Prompts come from the license-verified SFT manifest; the teacher's
            # license is pinned in policy.teachers and re-checked on its model card.
            "license_status": "usable",
            "shards": [shard_ref.to_dict()],
            "reports": [report_ref.to_dict(), leakage_ref.to_dict()],
            "language_statistics": report["language"],
            "dedup_statistics": dedup_stats,
            "leakage_statistics": {"status": leakage["status"], "hit_count": leakage["hit_count"]},
            "eval_suite_hash": suite_hash(),
            "policy_sha256": sha256_file(policy.path),
            "config_sha256": sha256_file(config.path),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        manifest_ref = _published(
            staging / "manifest.json", final / "manifest.json", "seqkd_manifest", repository_root
        )
        prompt_refs = [ref for ref in prompt_manifest["shards"] if ref["role"] == "sft_shard"]
        card = {
            "schema_version": 1,
            "run_id": run_id,
            "run_type": "seqkd_generate",
            "mode": "real",
            "status": "succeeded",
            "started_at": start["event"]["timestamp"],
            "finished_at": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "duration_seconds": time.monotonic() - started,
            "command": ["python", "-m", "boldt_posttrain.cli", "seqkd", "generate", "--real"],
            "git": provenance.collect_git("HEAD", root=repository_root),
            "policy": {"path": str(policy.path), "sha256": manifest["policy_sha256"]},
            "experiment": {
                "path": str(config.path),
                "sha256": manifest["config_sha256"],
                "resolved_sha256": sha256_bytes(canonical_json_bytes(config.document)),
            },
            "inputs": [dict(ref) for ref in prompt_refs],
            "outputs": [
                shard_ref.to_dict(),
                report_ref.to_dict(),
                leakage_ref.to_dict(),
                manifest_ref.to_dict(),
            ],
            "model": teacher_record,
            "data": manifest,
            "parameters": {**generation_parameters(settings), "max_model_len": max_model_len},
            "hardware": provenance.collect_hardware(),
            "environment": {
                **provenance.collect_environment(),
                "event_head": {
                    key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")
                },
            },
            "parents": [prompt_manifest["run_id"]],
            "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(teacher_record)),
            "error": None,
        }
        validate_run_card(card)
        run_staging.mkdir(parents=True)
        atomic_write_json(run_staging / "run_card.json", card)
        os.replace(staging, final)
        os.replace(run_staging, run_final)
        finish = events.append(
            "run_finished",
            run_id,
            {"status": "succeeded", "run_card_sha256": sha256_file(run_final / "run_card.json")},
        )
    except Exception:
        events.append("run_finished", run_id, {"status": "failed"})
        raise
    return {
        "status": "succeeded",
        "run_id": run_id,
        "manifest": str(final / "manifest.json"),
        "manifest_sha256": sha256_file(final / "manifest.json"),
        "rows_trainable": len(clean),
        "rejections": report["rejections"],
        "event_sequence": finish["sequence"],
    }


def verify_generation(
    settings: Mapping[str, Any],
    policy: Policy,
    *,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (manifest, run card) of the pinned generation run, or fail closed."""
    from .evaluation import suite_hash

    run_id = settings["generation_run"]
    if not run_id:
        raise SeqKDError("config.seqkd.generation_run is empty; run `pt seqkd generate` first")
    if not RUN_ID_RE.fullmatch(run_id):
        raise SeqKDError("config.seqkd.generation_run must be an exact run ID")
    card_path = outputs_root / "runs" / run_id / "run_card.json"
    if not card_path.is_file():
        raise SeqKDError(f"seqkd generation run {run_id} has no run card")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    try:
        validate_run_card(card)
    except ArtifactError as exc:
        raise SeqKDError(f"seqkd generation run card is invalid: {exc}") from exc
    if card["run_type"] != "seqkd_generate" or card["status"] != "succeeded":
        raise SeqKDError("pinned run is not a successful seqkd generation")
    if run_id not in _verified_event_runs(outputs_root):
        raise SeqKDError("seqkd generation run has no successful event-chain record")
    try:
        for ref in card["outputs"]:
            verify_artifact_ref(ref, root=repository_root)
    except ArtifactError as exc:
        raise SeqKDError(f"seqkd generation artifact verification failed: {exc}") from exc
    manifest_refs = [ref for ref in card["outputs"] if ref["role"] == "seqkd_manifest"]
    if len(manifest_refs) != 1:
        raise SeqKDError("seqkd generation must publish exactly one manifest")
    path = Path(manifest_refs[0]["path"])
    manifest = json.loads((path if path.is_absolute() else repository_root / path).read_text())
    if manifest.get("run_id") != run_id or manifest.get("status") != "trainable":
        raise SeqKDError("seqkd manifest identity or status is invalid")
    if manifest.get("eval_suite_hash") != suite_hash():
        raise SeqKDError("seqkd manifest eval-suite fingerprint is stale")
    if manifest.get("policy_sha256") != sha256_file(policy.path):
        raise SeqKDError("seqkd manifest was generated under a different policy")
    if manifest.get("leakage_statistics") != {"status": "clean", "hit_count": 0}:
        raise SeqKDError("seqkd manifest leakage gate is not clean")
    entry = policy_teacher(policy, settings["teacher"])
    recorded = manifest.get("teacher", {})
    if {key: recorded.get(key) for key in entry} != entry:
        raise SeqKDError("seqkd manifest teacher differs from the configured teacher")
    if manifest.get("generation") != generation_parameters(settings):
        raise SeqKDError("config.seqkd generation parameters changed since the pinned run")
    if [ref["path"] for ref in manifest.get("shards", [])] != [
        ref["path"] for ref in card["outputs"] if ref["role"] == "sft_shard"
    ]:
        raise SeqKDError("seqkd manifest shards differ from the run card outputs")
    return manifest, card
