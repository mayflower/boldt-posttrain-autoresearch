"""Verified adapter materialization and budgeted Mergekit search."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import config as config_module
from . import provenance
from .artifacts import (
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
from .policy import Policy, load_policy
from .resolver import CandidateRegistry, OUTPUTS, ResolvedModelRef, resolve_candidate
from .scoring import load_candidate_score

ROOT = Path(__file__).resolve().parents[2]


class MergeError(RuntimeError):
    """Merge inputs, materialization, execution, or budget failed closed."""


@dataclass(frozen=True)
class MergeInput:
    run_id: str
    resolved: ResolvedModelRef
    run_card: dict[str, Any]
    score: dict[str, Any]


def _artifact_path(ref: Mapping[str, Any], repository_root: Path) -> Path:
    path = Path(str(ref["path"]))
    return path if path.is_absolute() else repository_root / path


def eligible_input(
    run_id: str,
    policy: Policy,
    *,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> MergeInput:
    resolved = resolve_candidate(run_id, policy, outputs_root=outputs_root)
    card_path = outputs_root / "runs" / run_id / "run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if (
        card["mode"] != "real"
        or card["status"] != "succeeded"
        or card["run_type"] not in {"train_sft", "train_cpt", "train_preference", "merge"}
    ):
        raise MergeError("merge input is not a successful real training or merge run")
    if card["data"].get("license_status") != "usable" or card["data"].get("leakage_statistics") != {
        "status": "clean",
        "hit_count": 0,
    }:
        raise MergeError("merge input lacks usable license or clean leakage evidence")
    expected = policy.seed_model
    if (
        resolved.base_model != {"repo_id": expected["repo_id"], "revision": expected["revision"]}
        or resolved.tokenizer_sha256 != expected["tokenizer_sha256"]
        or resolved.chat_template_sha256 != expected["chat_template_sha256"]
        or resolved.model_config_sha256 != expected["model_config_sha256"]
        or resolved.architecture != expected["architecture"]
    ):
        raise MergeError("merge input compatibility differs from the exact seed")
    score, _ = load_candidate_score(
        run_id,
        policy,
        outputs_root=outputs_root,
        repository_root=repository_root,
    )
    if score["status"] != "passed" or not all(score["gates"].values()):
        raise MergeError("merge input has no passing verified real evaluation")
    return MergeInput(run_id, resolved, card, score)


def validate_input_set(inputs: list[MergeInput]) -> None:
    if len(inputs) < 2:
        raise MergeError("merge requires at least two eligible candidates")
    fingerprints = {
        (
            item.resolved.base_model["repo_id"],
            item.resolved.base_model["revision"],
            item.resolved.tokenizer_sha256,
            item.resolved.chat_template_sha256,
            item.resolved.model_config_sha256,
            item.resolved.architecture,
        )
        for item in inputs
    }
    if len(fingerprints) != 1:
        raise MergeError("merge candidates have different architecture or tokenizer fingerprints")


def _copy_tokenizer(source: str, revision: str | None, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    source_path = Path(source)
    if source_path.is_absolute():
        tokenizer_root = source_path
    else:
        tokenizer_root = Path(
            snapshot_download(
                source,
                revision=revision,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                    "tokenizer.model",
                ],
            )
        )
    copied = 0
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tokenizer.model",
    ):
        path = tokenizer_root / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied += 1
    if copied == 0:
        raise MergeError("exact seed tokenizer files are missing")


def _full_checkpoint_smoke(
    checkpoint: Path,
    *,
    expected_architecture: str,
    expected_tokenizer_sha256: str,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if sha256_file(checkpoint / "tokenizer.json") != expected_tokenizer_sha256:
        raise MergeError("materialized or merged tokenizer differs bytewise from the seed")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32
    )
    if model.__class__.__name__ != expected_architecture:
        raise MergeError("merged checkpoint architecture differs from candidates")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    encoded = tokenizer("Hallo", return_tensors="pt")
    with torch.inference_mode():
        model(**encoded)


def materialize_adapter(
    item: MergeInput,
    *,
    base_model_source: str,
    base_model_revision: str | None,
    output_root: Path,
    policy: Policy,
    allow_checkpoints: bool,
    repository_root: Path = ROOT,
    state_root: Path = OUTPUTS,
) -> tuple[Path, str]:
    if not allow_checkpoints:
        raise MergeError("adapter materialization requires explicit checkpoint permission")
    if item.resolved.kind != "peft_adapter" or item.resolved.artifact is None:
        raise MergeError("materialization accepts only a verified PEFT adapter")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    run_id = new_run_id("materialize")
    staging, final = output_root / ".staging" / run_id, output_root / run_id
    run_staging, run_final = state_root / "runs/.staging" / run_id, state_root / "runs" / run_id
    staging.mkdir(parents=True)
    run_staging.mkdir(parents=True)
    events = EventLog(state_root)
    start = events.append("run_started", run_id, {"run_type": "merge"})
    local = Path(base_model_source).is_absolute()
    base = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        revision=base_model_revision,
        local_files_only=local,
        dtype=torch.float32,
    )
    adapter_path = _artifact_path(item.resolved.artifact, repository_root)
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(staging, safe_serialization=True)
    _copy_tokenizer(base_model_source, base_model_revision, staging)
    _full_checkpoint_smoke(
        staging,
        expected_architecture=item.resolved.architecture,
        expected_tokenizer_sha256=item.resolved.tokenizer_sha256,
    )
    measured = ArtifactRef.from_path(
        staging,
        role="full_checkpoint",
        media_type="application/vnd.boldt.transformers-checkpoint",
    )
    try:
        stored = final.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        stored = str(final.resolve())
    output_ref = ArtifactRef(
        stored,
        measured.kind,
        measured.role,
        measured.sha256,
        measured.size_bytes,
        measured.media_type,
    )
    model = item.resolved.to_dict()
    model["kind"] = "local_full_checkpoint"
    model["artifact"] = output_ref.to_dict()
    model["source_run_id"] = run_id
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    card = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": "merge",
        "mode": "real",
        "status": "succeeded",
        "started_at": start["event"]["timestamp"],
        "finished_at": now,
        "duration_seconds": 0.0,
        "command": ["python", "-m", "boldt_posttrain.cli", "merge", "materialize"],
        "git": provenance.collect_git("HEAD", root=repository_root),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "none",
            "sha256": sha256_bytes(b"null"),
            "resolved_sha256": sha256_bytes(b"null"),
        },
        "inputs": [item.resolved.artifact],
        "outputs": [output_ref.to_dict()],
        "model": model,
        "data": item.run_card["data"],
        "parameters": {"operation": "peft_merge_and_unload"},
        "hardware": provenance.collect_hardware(),
        "environment": {
            **provenance.collect_environment(),
            "event_head": {
                key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")
            },
        },
        "parents": [item.run_id],
        "compatibility_fingerprint": item.run_card["compatibility_fingerprint"],
        "error": None,
    }
    validate_run_card(card)
    atomic_write_json(run_staging / "run_card.json", card)
    os.replace(staging, final)
    os.replace(run_staging, run_final)
    events.append(
        "run_finished",
        run_id,
        {"status": "succeeded", "run_card_sha256": sha256_file(run_final / "run_card.json")},
    )
    return final, run_id


def merge_configuration(
    method: str,
    model_paths: list[Path],
    *,
    dtype: str,
    parameters: Mapping[str, Any],
    base_model: str | None = None,
) -> str:
    from mergekit.config import MergeConfiguration

    if method not in {"linear", "slerp", "ties", "dare_ties"}:
        raise MergeError(f"unsupported merge method {method}")
    paths = [str(path.resolve()) for path in model_paths]
    if len(paths) < 2:
        raise MergeError("merge configuration requires two or more models")
    if method == "linear":
        weights = parameters.get("weights") or [1.0 / len(paths)] * len(paths)
        if len(weights) != len(paths):
            raise MergeError("linear weight count differs from model count")
        document: dict[str, Any] = {
            "models": [
                {"model": path, "parameters": {"weight": float(weight)}}
                for path, weight in zip(paths, weights, strict=True)
            ],
            "merge_method": method,
            "dtype": dtype,
            "tokenizer_source": paths[0],
        }
    elif method == "slerp":
        if len(paths) != 2:
            raise MergeError("SLERP requires exactly two models")
        document = {
            "models": [{"model": path} for path in paths],
            "merge_method": method,
            "base_model": paths[0],
            "parameters": {"t": float(parameters.get("t", 0.5))},
            "dtype": dtype,
            "tokenizer_source": paths[0],
        }
    else:
        if base_model is None:
            raise MergeError(f"{method} requires the exact seed base model")
        density = float(parameters.get("density", 0.5))
        weight = float(parameters.get("weight", 1.0))
        document = {
            "models": [
                {"model": path, "parameters": {"weight": weight, "density": density}}
                for path in paths
            ],
            "merge_method": method,
            "base_model": base_model,
            "dtype": dtype,
            "tokenizer_source": base_model,
        }
    return MergeConfiguration.model_validate(document).to_yaml() + "\n"


def merge_parameter_grid(
    methods: list[str], parameters: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    combinations: list[tuple[str, dict[str, Any]]] = []
    for method in methods:
        configured = parameters.get(method, {})
        variants = configured if isinstance(configured, list) else [configured]
        if not variants:
            raise MergeError(f"merge parameter grid is empty for {method}")
        for variant in variants:
            if not isinstance(variant, dict):
                raise MergeError(f"merge parameters for {method} must be objects")
            combinations.append((method, dict(variant)))
    return combinations


def validate_merge_parameters(
    method: str,
    parameters: Mapping[str, Any],
    *,
    model_count: int,
    bounds: Mapping[str, Any],
) -> None:
    allowed = {
        "linear": {"weights"},
        "slerp": {"t"},
        "ties": {"weight", "density"},
        "dare_ties": {"weight", "density"},
    }[method]
    unknown = set(parameters) - allowed
    if unknown:
        raise MergeError(f"unknown {method} merge parameters: {sorted(unknown)}")
    if method == "linear":
        weights = parameters.get("weights", [1.0 / model_count] * model_count)
        if not isinstance(weights, list) or len(weights) != model_count:
            raise MergeError("linear weight count differs from model count")
        values = [float(value) for value in weights]
        if any(
            not math.isfinite(value) or not bounds["weight_min"] <= value <= bounds["weight_max"]
            for value in values
        ):
            raise MergeError("linear merge weight is outside policy bounds")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise MergeError("linear merge weights must sum to one")
        return
    if method == "slerp":
        value = float(parameters.get("t", 0.5))
        if not math.isfinite(value) or not bounds["weight_min"] <= value <= bounds["weight_max"]:
            raise MergeError("SLERP t is outside policy bounds")
        return
    weight = float(parameters.get("weight", 1.0))
    density = float(parameters.get("density", 0.5))
    if not math.isfinite(weight) or not bounds["weight_min"] <= weight <= bounds["weight_max"]:
        raise MergeError(f"{method} weight is outside policy bounds")
    if not math.isfinite(density) or not bounds["density_min"] <= density <= bounds["density_max"]:
        raise MergeError(f"{method} density is outside policy bounds")


def execute_merge(
    *,
    method: str,
    model_paths: list[Path],
    input_refs: list[Mapping[str, Any]],
    parent_run_ids: list[str],
    model_metadata: Mapping[str, Any],
    data_metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    dtype: str,
    output_checkpoint_root: Path,
    output_merge_root: Path,
    state_root: Path,
    policy: Policy,
    allow_checkpoints: bool,
    use_gpu: bool,
    deadline: float,
    repository_root: Path = ROOT,
    base_model: str | None = None,
) -> dict[str, Any]:
    if not allow_checkpoints:
        raise MergeError("merge execution requires explicit checkpoint permission")
    if time.monotonic() >= deadline:
        raise MergeError("merge budget exhausted before candidate start")
    executable = shutil.which("mergekit-yaml")
    if executable is None:
        raise MergeError("mergekit-yaml executable is unavailable")
    run_id = new_run_id("merge")
    checkpoint_staging = output_checkpoint_root / ".staging" / run_id
    checkpoint_final = output_checkpoint_root / run_id
    merge_staging = output_merge_root / ".staging" / run_id
    merge_final = output_merge_root / run_id
    run_staging, run_final = state_root / "runs/.staging" / run_id, state_root / "runs" / run_id
    checkpoint_staging.parent.mkdir(parents=True, exist_ok=True)
    merge_staging.mkdir(parents=True)
    run_staging.mkdir(parents=True)
    events = EventLog(state_root)
    started = time.monotonic()
    start = events.append("run_started", run_id, {"run_type": "merge"})
    yaml = merge_configuration(
        method,
        model_paths,
        dtype=dtype,
        parameters=parameters,
        base_model=base_model,
    )
    config_path = merge_staging / "mergekit.yaml"
    atomic_write_bytes(config_path, yaml.encode())
    command = [
        executable,
        str(config_path),
        str(checkpoint_staging),
        "--safe-serialization",
        "--copy-tokenizer",
        "--no-write-model-card",
        "--random-seed",
        "1729",
    ]
    command.append("--cuda" if use_gpu else "--no-cuda")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise MergeError("merge budget exhausted before Mergekit execution")
    try:
        process = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as exc:
        raise MergeError("Mergekit exceeded the total merge deadline") from exc
    atomic_write_bytes(merge_staging / "stdout.log", process.stdout.encode())
    atomic_write_bytes(merge_staging / "stderr.log", process.stderr.encode())
    if process.returncode != 0:
        raise MergeError(
            f"Mergekit failed with exit {process.returncode}: {process.stderr[-2000:]}"
        )
    _full_checkpoint_smoke(
        checkpoint_staging,
        expected_architecture=str(model_metadata["architecture"]),
        expected_tokenizer_sha256=str(model_metadata["tokenizer_sha256"]),
    )
    checkpoint_measured = ArtifactRef.from_path(
        checkpoint_staging,
        role="merged_checkpoint",
        media_type="application/vnd.boldt.transformers-checkpoint",
    )
    config_measured = ArtifactRef.from_path(
        config_path,
        role="mergekit_config",
        media_type="application/yaml",
    )

    def published(measured: ArtifactRef, destination: Path) -> ArtifactRef:
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

    checkpoint_ref = published(checkpoint_measured, checkpoint_final)
    config_ref = published(config_measured, merge_final / "mergekit.yaml")
    merged_model = {
        **dict(model_metadata),
        "kind": "merged_checkpoint",
        "requested": run_id,
        "artifact": checkpoint_ref.to_dict(),
        "source_run_id": run_id,
    }
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    card = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": "merge",
        "mode": "real",
        "status": "succeeded",
        "started_at": start["event"]["timestamp"],
        "finished_at": now,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "git": provenance.collect_git("HEAD", root=repository_root),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "in-memory",
            "sha256": sha256_bytes(canonical_json_bytes(parameters)),
            "resolved_sha256": sha256_bytes(canonical_json_bytes(parameters)),
        },
        "inputs": [dict(item) for item in input_refs],
        "outputs": [checkpoint_ref.to_dict(), config_ref.to_dict()],
        "model": merged_model,
        "data": dict(data_metadata),
        "parameters": {"method": method, **dict(parameters)},
        "hardware": provenance.collect_hardware(),
        "environment": {
            **provenance.collect_environment(),
            "mergekit_exit_code": process.returncode,
            "event_head": {
                key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")
            },
        },
        "parents": list(parent_run_ids),
        "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(model_metadata)),
        "error": None,
    }
    validate_run_card(card)
    atomic_write_json(run_staging / "run_card.json", card)
    os.replace(checkpoint_staging, checkpoint_final)
    os.replace(merge_staging, merge_final)
    os.replace(run_staging, run_final)
    events.append(
        "run_finished",
        run_id,
        {"status": "succeeded", "run_card_sha256": sha256_file(run_final / "run_card.json")},
    )
    CandidateRegistry(state_root).rebuild(policy)
    return {
        "status": "succeeded",
        "run_id": run_id,
        "method": method,
        "checkpoint": str(checkpoint_final),
    }


def run_search(
    *,
    candidate_ids: list[str],
    methods: list[str],
    parameters: Mapping[str, Any],
    dtype: str,
    policy: Policy,
    allow_checkpoints: bool,
    allow_gpu: bool,
    budget_minutes: float,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    if not candidate_ids:
        raise MergeError("merge candidate space is empty")
    if any(method not in policy.document["merge"]["allowed_methods"] for method in methods):
        raise MergeError("experiment requests a merge method outside policy")
    inputs = [
        eligible_input(
            run_id,
            policy,
            outputs_root=outputs_root,
            repository_root=repository_root,
        )
        for run_id in candidate_ids
    ]
    validate_input_set(inputs)
    deadline = time.monotonic() + budget_minutes * 60
    paths: list[Path] = []
    refs: list[Mapping[str, Any]] = []
    for item in inputs:
        if time.monotonic() >= deadline:
            raise MergeError("merge budget exhausted during input materialization")
        if item.resolved.kind == "peft_adapter":
            path, materialization_id = materialize_adapter(
                item,
                base_model_source=policy.seed_model["repo_id"],
                base_model_revision=policy.seed_model["revision"],
                output_root=outputs_root / "checkpoints",
                policy=policy,
                allow_checkpoints=allow_checkpoints,
                repository_root=repository_root,
                state_root=outputs_root,
            )
            materialized_card = json.loads(
                (outputs_root / "runs" / materialization_id / "run_card.json").read_text()
            )
            ref = next(
                value
                for value in materialized_card["outputs"]
                if value["role"] == "full_checkpoint"
            )
        else:
            assert item.resolved.artifact is not None
            path = _artifact_path(item.resolved.artifact, repository_root)
            ref = item.resolved.artifact
        paths.append(path)
        refs.append(ref)
    search_space = merge_parameter_grid(methods, parameters)
    if len(search_space) > policy.document["merge"]["max_candidates"]:
        raise MergeError("merge search exceeds protected candidate limit")
    results = []
    for method, method_parameters in search_space:
        if time.monotonic() >= deadline:
            break
        bounds = policy.document["merge"]["parameter_bounds"]
        validate_merge_parameters(
            method,
            method_parameters,
            model_count=len(paths),
            bounds=bounds,
        )
        results.append(
            execute_merge(
                method=method,
                model_paths=paths,
                input_refs=refs,
                parent_run_ids=[item.run_id for item in inputs],
                model_metadata=inputs[0].resolved.to_dict(),
                data_metadata=inputs[0].run_card["data"],
                parameters=method_parameters,
                dtype=dtype,
                output_checkpoint_root=outputs_root / "checkpoints",
                output_merge_root=outputs_root / "merge",
                state_root=outputs_root,
                policy=policy,
                allow_checkpoints=allow_checkpoints,
                use_gpu=allow_gpu,
                deadline=deadline,
                repository_root=repository_root,
                base_model=f"{policy.seed_model['repo_id']}@{policy.seed_model['revision']}",
            )
        )
    if not results:
        raise MergeError("merge search produced no candidate before deadline")
    return {
        "status": "succeeded",
        "candidates": results,
        "budget_exhausted": len(results) < len(search_space),
    }


def run_cli(args) -> tuple[dict[str, Any], int]:
    policy = load_policy()
    config = config_module.load_experiment(ROOT / args.config)
    merge = config.document["merge"]
    result = run_search(
        candidate_ids=merge["inputs"],
        methods=merge["methods"],
        parameters=merge["parameters"],
        dtype=merge["dtype"],
        policy=policy,
        allow_checkpoints=args.allow_checkpoints,
        allow_gpu=args.allow_gpu,
        budget_minutes=args.budget_minutes or config.document["resources"]["budget_minutes"],
    )
    return result, 0
