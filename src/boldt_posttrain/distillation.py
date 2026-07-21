"""Offline local teacher distillation with immutable data and student lineage."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

from . import config as config_module
from . import provenance
from .artifacts import (
    RUN_ID_RE,
    ArtifactRef,
    EventLog,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
)
from .data_pipeline import (
    LanguageIdentifier,
    deduplicate,
    leakage_filter,
    normalize_license,
    normalize_row,
    verify_data_manifest,
)
from .evaluation import load_transformers_model, suite_hash
from .policy import Policy, load_policy
from .resolver import OUTPUTS, ResolvedModelRef, resolve_model
from .training import train_adapter

ROOT = Path(__file__).resolve().parents[2]


class DistillationError(RuntimeError):
    """Teacher, generated data, or student training failed a protected gate."""


def _published_ref(
    source: Path,
    destination: Path,
    *,
    role: str,
    media_type: str,
    repository_root: Path,
) -> ArtifactRef:
    measured = ArtifactRef.from_path(source, role=role, media_type=media_type)
    try:
        stored = destination.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        stored = str(destination.resolve())
    return ArtifactRef(
        stored,
        measured.kind,
        measured.role,
        measured.sha256,
        measured.size_bytes,
        measured.media_type,
    )


def _teacher_license(
    teacher: ResolvedModelRef, policy: Policy, declared_local_license: str | None
) -> str:
    if teacher.kind == "hub_model":
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            teacher.base_model["repo_id"], revision=teacher.base_model["revision"]
        )
        card = getattr(info, "card_data", None)
        raw = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    else:
        raw = declared_local_license
        if raw is None:
            raise DistillationError("local teachers require --teacher-license")
    license_id = normalize_license(raw)
    if license_id not in policy.document["data"]["allowed_licenses"]:
        raise DistillationError("teacher license is unknown or forbidden by policy")
    return license_id


def extract_prompts(
    manifest: Mapping[str, Any], *, repository_root: Path = ROOT, maximum: int
) -> list[str]:
    prompts: list[str] = []
    for ref in manifest["shards"]:
        if ref["role"] not in {"sft_shard", "cpt_shard"}:
            continue
        path = Path(ref["path"])
        path = path if path.is_absolute() else repository_root / path
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw:
                continue
            row = json.loads(raw)
            if row["type"] == "sft":
                users = [item["content"] for item in row["messages"] if item["role"] == "user"]
                if users:
                    prompts.append(users[-1])
            elif row["type"] == "cpt":
                prompts.append(row["text"])
            if len(prompts) >= maximum:
                return prompts
    if not prompts:
        raise DistillationError("verified source manifest contains no distillation prompts")
    return prompts


def generate_teacher_outputs(
    teacher: ResolvedModelRef,
    prompts: list[str],
    *,
    device: str,
    generation: Mapping[str, int],
) -> list[dict[str, Any]]:
    import torch

    minimum = int(generation["min_new_tokens"])
    maximum = int(generation["max_new_tokens"])
    if minimum < 1 or maximum < minimum:
        raise DistillationError("teacher generation token bounds are invalid")
    random.seed(42)
    torch.manual_seed(42)
    model, tokenizer = load_transformers_model(teacher, device=device)
    if not tokenizer.chat_template:
        raise DistillationError("teacher tokenizer has no chat template")
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    temperature=None,
                    min_new_tokens=minimum,
                    max_new_tokens=maximum,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            records.append(
                {
                    "prompt_id": f"teacher-{index:08d}",
                    "prompt": prompt,
                    "output": output,
                    "error": None,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "prompt_id": f"teacher-{index:08d}",
                    "prompt": prompt,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records


def distill_and_train(
    *,
    teacher: ResolvedModelRef,
    teacher_license: str,
    student_model_source: str,
    student_model_revision: str | None,
    prompts: list[str],
    output_data_root: Path,
    output_checkpoint_root: Path,
    policy: Policy,
    training: Mapping[str, Any],
    generation: Mapping[str, int],
    target_modules: list[str],
    device: str,
    qlora: bool,
    allow_checkpoints: bool,
    budget_minutes: float,
    repository_root: Path = ROOT,
    language_identifier: Any | None = None,
) -> dict[str, Any]:
    if not allow_checkpoints:
        raise DistillationError("distillation requires explicit checkpoint permission")
    if not prompts:
        raise DistillationError("distillation prompt set is empty")
    started = time.monotonic()
    deadline = started + budget_minutes * 60
    data_run_id = new_run_id("data-distill")
    staging = output_data_root / ".staging" / data_run_id
    final = output_data_root / data_run_id
    staging.mkdir(parents=True)
    state_root = output_data_root.parent
    events = EventLog(state_root)
    start_event = events.append("run_started", data_run_id, {"run_type": "distill"})
    records = generate_teacher_outputs(teacher, prompts, device=device, generation=generation)
    language = language_identifier or LanguageIdentifier(policy)
    normalized: list[dict[str, Any]] = []
    rejections: dict[str, int] = {"generation_error": 0, "empty": 0, "language": 0}
    source = {
        "dataset_id": f"distillation/{teacher.base_model['repo_id'].replace('/', '--')}",
        "revision": teacher.base_model["revision"],
        "config": "offline-teacher-v1",
        "split": "train",
        "license": teacher_license,
    }
    for record in records:
        if record["error"]:
            rejections["generation_error"] += 1
            continue
        if not record["output"]:
            rejections["empty"] += 1
            continue
        german, confidence = language.check(record["output"])
        record["language_confidence"] = confidence
        if not german:
            rejections["language"] += 1
            continue
        normalized.append(
            normalize_row(
                {"prompt": record["prompt"], "response": record["output"]},
                source,
                record["prompt_id"],
            )
        )
    deduplicated, dedup_stats = deduplicate(
        normalized, float(policy.document["data"]["near_dedup_jaccard"])
    )
    clean, leakage = leakage_filter(deduplicated, policy)
    if leakage["status"] != "clean":
        raise DistillationError("teacher outputs failed benchmark-leakage gate")
    if not clean:
        raise DistillationError(
            "no teacher output passed language, deduplication, and leakage gates"
        )
    raw_path = staging / "teacher_generations.jsonl"
    shard_path = staging / "train_sft-00000-of-00001.jsonl"
    atomic_write_bytes(raw_path, b"".join(canonical_json_bytes(row) + b"\n" for row in records))
    atomic_write_bytes(shard_path, b"".join(canonical_json_bytes(row) + b"\n" for row in clean))
    quality = {
        "schema_version": 1,
        "prompts": len(prompts),
        "generated": len(records),
        "trainable": len(clean),
        "rejections": rejections,
        "dedup": dedup_stats,
    }
    atomic_write_json(staging / "quality_report.json", quality)
    atomic_write_json(staging / "leakage_report.json", leakage)
    raw_ref = _published_ref(
        raw_path,
        final / raw_path.name,
        role="teacher_generations",
        media_type="application/jsonl",
        repository_root=repository_root,
    )
    shard_ref = _published_ref(
        shard_path,
        final / shard_path.name,
        role="sft_shard",
        media_type="application/jsonl",
        repository_root=repository_root,
    )
    quality_ref = _published_ref(
        staging / "quality_report.json",
        final / "quality_report.json",
        role="quality_report",
        media_type="application/json",
        repository_root=repository_root,
    )
    leakage_ref = _published_ref(
        staging / "leakage_report.json",
        final / "leakage_report.json",
        role="leakage_report",
        media_type="application/json",
        repository_root=repository_root,
    )
    manifest = {
        "schema_version": 1,
        "run_id": data_run_id,
        "status": "trainable",
        "sources": [source],
        "shards": [shard_ref.to_dict()],
        "reports": [quality_ref.to_dict(), leakage_ref.to_dict(), raw_ref.to_dict()],
        "license_status": "usable",
        "language_statistics": {
            "checked": len(records) - rejections["generation_error"] - rejections["empty"],
            "rejected": rejections["language"],
        },
        "dedup_statistics": dedup_stats,
        "leakage_statistics": {"status": "clean", "hit_count": 0},
        "eval_suite_hash": suite_hash(),
        "policy_sha256": sha256_file(policy.path),
        "teacher": teacher.to_dict(),
        "generation": dict(generation),
    }
    atomic_write_json(staging / "manifest.json", manifest)
    manifest_ref = _published_ref(
        staging / "manifest.json",
        final / "manifest.json",
        role="data_manifest",
        media_type="application/json",
        repository_root=repository_root,
    )
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    teacher_inputs = [teacher.artifact] if teacher.artifact is not None else []
    card = {
        "schema_version": 1,
        "run_id": data_run_id,
        "run_type": "distill",
        "mode": "real",
        "status": "succeeded",
        "started_at": start_event["event"]["timestamp"],
        "finished_at": now,
        "duration_seconds": time.monotonic() - started,
        "command": ["python", "-m", "boldt_posttrain.cli", "distill", "--real"],
        "git": provenance.collect_git("HEAD", root=repository_root),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "in-memory",
            "sha256": sha256_bytes(canonical_json_bytes(generation)),
            "resolved_sha256": sha256_bytes(canonical_json_bytes(generation)),
        },
        "inputs": teacher_inputs,
        "outputs": [
            raw_ref.to_dict(),
            shard_ref.to_dict(),
            quality_ref.to_dict(),
            leakage_ref.to_dict(),
            manifest_ref.to_dict(),
        ],
        "model": {"teacher": teacher.to_dict()},
        "data": manifest,
        "parameters": {"generation": dict(generation)},
        "hardware": provenance.collect_hardware(),
        "environment": {
            **provenance.collect_environment(),
            "event_head": {
                key: start_event[key] for key in ("sequence", "last_event_hash", "log_sha256")
            },
        },
        "parents": [teacher.source_run_id] if teacher.source_run_id else [],
        "compatibility_fingerprint": sha256_bytes(
            canonical_json_bytes({"teacher": teacher.to_dict(), "suite": suite_hash()})
        ),
        "error": None,
    }
    validate_run_card(card)
    atomic_write_json(staging / "run_card.json", card)
    os.replace(staging, final)
    events.append(
        "run_finished",
        data_run_id,
        {"status": "succeeded", "run_card_sha256": sha256_file(final / "run_card.json")},
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DistillationError("distillation budget exhausted before student training")
    from datasets import Dataset

    student = train_adapter(
        kind="sft",
        model_source=student_model_source,
        revision=student_model_revision,
        dataset=Dataset.from_list([{"messages": row["messages"]} for row in clean]),
        output_root=output_checkpoint_root,
        policy=policy,
        experiment=training,
        target_modules=target_modules,
        device=device,
        qlora=qlora,
        allow_checkpoints=allow_checkpoints,
        budget_minutes=remaining / 60,
        repository_root=repository_root,
        input_artifacts=[manifest_ref.to_dict()],
        parent_run_ids=[data_run_id],
        lineage={
            "teacher": teacher.to_dict(),
            "teacher_data_run_id": data_run_id,
            "teacher_data_sha256": manifest_ref.sha256,
        },
        data_metadata=manifest,
    )
    return {
        "status": student["status"],
        "teacher_data_run_id": data_run_id,
        "student_run_id": student["run_id"],
        "teacher_data_manifest": str(final / "manifest.json"),
        "student_checkpoint": student["checkpoint"],
    }


def run_cli(args) -> tuple[dict[str, Any], int]:
    import torch

    if not torch.cuda.is_available():
        raise DistillationError("real distillation requires CUDA and cannot fall back to CPU")
    policy = load_policy()
    config = config_module.load_experiment(ROOT / args.config)
    source_manifest = verify_data_manifest(OUTPUTS / "data", policy)
    prompts = extract_prompts(
        source_manifest,
        maximum=config.document["distillation"]["max_prompts"],
    )
    if RUN_ID_RE.fullmatch(args.teacher):
        teacher = resolve_model(policy=policy, candidate=args.teacher)
    else:
        teacher_path = Path(args.teacher)
        external = (teacher_path.resolve().parent,) if teacher_path.exists() else ()
        teacher = resolve_model(policy=policy, model=args.teacher, external_roots=external)
    license_id = _teacher_license(teacher, policy, args.teacher_license)
    result = distill_and_train(
        teacher=teacher,
        teacher_license=license_id,
        student_model_source=policy.seed_model["repo_id"],
        student_model_revision=policy.seed_model["revision"],
        prompts=prompts,
        output_data_root=OUTPUTS / "data",
        output_checkpoint_root=OUTPUTS / "checkpoints",
        policy=policy,
        training=config.document["training"],
        generation=config.document["distillation"],
        target_modules=config.document["training"]["target_modules"],
        device="cuda:0",
        qlora=config.document["training"]["method"] == "qlora",
        allow_checkpoints=args.allow_checkpoints,
        budget_minutes=args.budget_minutes or config.document["resources"]["budget_minutes"],
    )
    return result, 0 if result["status"] == "succeeded" else 1
