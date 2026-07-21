"""Real DPO, KTO, and experimental ORPO training on the pinned TRL API."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from . import config as config_module
from . import provenance
from .artifacts import (
    ArtifactRef,
    EventLog,
    atomic_write_json,
    canonical_json_bytes,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
)
from .data_pipeline import verify_data_manifest
from .policy import Policy, load_policy
from .resolver import CandidateRegistry, OUTPUTS
from .training import (
    DeadlineCallback,
    _checkpoint_smoke,
    collect_model_metadata,
    create_model_and_tokenizer,
    validate_target_modules,
    validate_tokenizer,
)

ROOT = Path(__file__).resolve().parents[2]


class PreferenceError(RuntimeError):
    """Preference data or one of the explicit trainer paths is invalid."""


def response_suppression_diagnostics(
    model, tokenizer, prompts: list[str], *, max_new_tokens: int
) -> dict[str, Any]:
    import torch

    refusal_terms = ("kann ich nicht", "dabei kann ich", "nicht helfen", "ablehnen")
    lengths: list[int] = []
    empty = 0
    refusals = 0
    model.eval()
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer_ids = generated[0, encoded["input_ids"].shape[1] :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        lengths.append(len(answer_ids))
        empty += int(not answer)
        refusals += int(any(term in answer.casefold() for term in refusal_terms))
    return {
        "count": len(prompts),
        "mean_response_tokens": sum(lengths) / len(lengths),
        "empty_rate": empty / len(prompts),
        "refusal_rate": refusals / len(prompts),
    }


def validate_preference_rows(
    rows: list[Mapping[str, Any]], tokenizer, settings: Mapping[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise PreferenceError("preference dataset is empty")
    lengths: list[dict[str, int]] = []
    for index, row in enumerate(rows):
        if set(("prompt", "chosen", "rejected")) - set(row):
            raise PreferenceError(f"preference row {index} is incomplete")
        prompt, chosen, rejected = (
            str(row[key]).strip() for key in ("prompt", "chosen", "rejected")
        )
        if not prompt or not chosen or not rejected or chosen == rejected:
            raise PreferenceError(f"preference row {index} has empty or identical answers")
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        chosen_length = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
        rejected_length = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
        if prompt_length > settings["max_prompt_length"]:
            raise PreferenceError(f"preference row {index} prompt exceeds hard maximum")
        if max(chosen_length, rejected_length) > settings["max_completion_length"]:
            raise PreferenceError(f"preference row {index} completion exceeds hard maximum")
        ratio = max(chosen_length, rejected_length) / max(1, min(chosen_length, rejected_length))
        if ratio > settings["length_ratio_max"]:
            raise PreferenceError(f"preference row {index} length ratio exceeds policy")
        lengths.append(
            {"prompt": prompt_length, "chosen": chosen_length, "rejected": rejected_length}
        )
    return {
        "count": len(rows),
        "mean_prompt_tokens": sum(item["prompt"] for item in lengths) / len(lengths),
        "mean_chosen_tokens": sum(item["chosen"] for item in lengths) / len(lengths),
        "mean_rejected_tokens": sum(item["rejected"] for item in lengths) / len(lengths),
    }


def preference_dataset(rows: list[Mapping[str, Any]], method: str):
    from datasets import Dataset

    if method == "kto":
        converted = []
        for row in rows:
            converted.extend(
                [
                    {"prompt": row["prompt"], "completion": row["chosen"], "label": True},
                    {"prompt": row["prompt"], "completion": row["rejected"], "label": False},
                ]
            )
        labels = {item["label"] for item in converted}
        if labels != {True, False}:
            raise PreferenceError("KTO requires both desirable and undesirable examples")
        return Dataset.from_list(converted)
    return Dataset.from_list(
        [
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in rows
        ]
    )


def _trainer(method: str):
    if method == "dpo":
        from trl import DPOConfig, DPOTrainer

        return DPOTrainer, DPOConfig
    if method == "kto":
        from trl import KTOConfig, KTOTrainer

        return KTOTrainer, KTOConfig
    if method == "orpo":
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        return ORPOTrainer, ORPOConfig
    raise PreferenceError(f"unsupported preference method {method}")


def train_preference_adapter(
    *,
    method: str,
    model_source: str,
    revision: str | None,
    rows: list[Mapping[str, Any]],
    output_root: Path,
    policy: Policy,
    training: Mapping[str, Any],
    preference: Mapping[str, Any],
    target_modules: list[str],
    device: str,
    qlora: bool,
    allow_checkpoints: bool,
    budget_minutes: float,
    repository_root: Path = ROOT,
    data_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not allow_checkpoints:
        raise PreferenceError("checkpoint writes require explicit permission")
    if method not in {"dpo", "kto", "orpo"}:
        raise PreferenceError("method must be dpo, kto, or orpo")
    from peft import LoraConfig

    model, tokenizer = create_model_and_tokenizer(
        model_source,
        revision=revision,
        qlora=qlora,
        gradient_checkpointing=training["gradient_checkpointing"],
        policy=policy,
        device=device,
    )
    validate_tokenizer(tokenizer, training["context_length"], model)
    model_metadata = collect_model_metadata(model_source, revision, model, policy)
    validate_target_modules(model, target_modules)
    diagnostics = validate_preference_rows(rows, tokenizer, preference)
    if method == "kto" and training["per_device_batch_size"] <= 1:
        raise PreferenceError("KTO requires per_device_batch_size greater than one")
    probe_prompts = [str(row["prompt"]) for row in rows[: min(16, len(rows))]]
    suppression_before = response_suppression_diagnostics(
        model,
        tokenizer,
        probe_prompts,
        max_new_tokens=min(64, preference["max_completion_length"]),
    )
    dataset = preference_dataset(rows, method)
    trainer_class, config_class = _trainer(method)
    run_id = new_run_id(f"train-{method}")
    staging, final = output_root / ".staging" / run_id, output_root / run_id
    state_root = output_root.parent
    run_staging = state_root / "runs/.staging" / run_id
    run_final = state_root / "runs" / run_id
    events = EventLog(state_root)
    start_event = events.append("run_started", run_id, {"run_type": "train_preference"})
    staging.mkdir(parents=True)
    started = time.monotonic()
    callback = DeadlineCallback(started + budget_minutes * 60)
    common: dict[str, Any] = {
        "output_dir": str(staging / "trainer"),
        "per_device_train_batch_size": training["per_device_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "num_train_epochs": training["num_train_epochs"],
        "max_steps": training["max_steps"],
        "warmup_ratio": training["warmup_ratio"],
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "seed": training["seed"],
        "data_seed": training["seed"],
        "use_cpu": device == "cpu",
        "bf16": device != "cpu",
        "gradient_checkpointing": training["gradient_checkpointing"],
        "max_length": min(
            training["context_length"],
            preference["max_prompt_length"] + preference["max_completion_length"],
        ),
        "beta": preference["beta"],
    }
    if method == "dpo":
        common["loss_type"] = [preference["loss_type"]]
    elif method == "orpo":
        common["max_completion_length"] = preference["max_completion_length"]
    args = config_class(**common)
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=training["lora_r"],
        lora_alpha=training["lora_alpha"],
        lora_dropout=training["lora_dropout"],
        target_modules=target_modules,
        bias="none",
    )
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[callback],
    )
    result = trainer.train()
    suppression_after = response_suppression_diagnostics(
        trainer.model,
        tokenizer,
        probe_prompts,
        max_new_tokens=min(64, preference["max_completion_length"]),
    )
    trainer.save_model(staging)
    tokenizer.save_pretrained(staging)
    if not (staging / "chat_template.jinja").exists() and tokenizer.chat_template:
        (staging / "chat_template.jinja").write_text(tokenizer.chat_template)
    _checkpoint_smoke(staging, model_source, revision)
    status = "budget_exhausted" if callback.exhausted else "succeeded"
    checkpoint = ArtifactRef.from_path(
        staging,
        role="adapter_checkpoint",
        media_type="application/vnd.boldt.peft-adapter",
    )
    try:
        stored = final.relative_to(repository_root).as_posix()
    except ValueError:
        stored = str(final)
    checkpoint = ArtifactRef(
        stored,
        checkpoint.kind,
        checkpoint.role,
        checkpoint.sha256,
        checkpoint.size_bytes,
        checkpoint.media_type,
    )
    metrics = {
        "train_loss": float(result.training_loss),
        "steps_completed": int(trainer.state.global_step),
        "preference_diagnostics": diagnostics,
        "response_suppression_before": suppression_before,
        "response_suppression_after": suppression_after,
        "positive_count": len(rows) if method == "kto" else None,
        "negative_count": len(rows) if method == "kto" else None,
        "log_history": trainer.state.log_history,
        "wall_clock_seconds": time.monotonic() - started,
        "stop_reason": status,
    }
    run_staging.mkdir(parents=True)
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    parameters = {"method": method, "training": dict(training), "preference": dict(preference)}
    card = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": "train_preference",
        "mode": "real",
        "status": status,
        "started_at": start_event["event"]["timestamp"],
        "finished_at": now,
        "duration_seconds": metrics["wall_clock_seconds"],
        "command": [
            "python",
            "-m",
            "boldt_posttrain.cli",
            "train",
            "preference",
            "--method",
            method,
            "--real",
        ],
        "git": provenance.collect_git("HEAD", root=repository_root),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "in-memory",
            "sha256": sha256_bytes(canonical_json_bytes(parameters)),
            "resolved_sha256": sha256_bytes(canonical_json_bytes(parameters)),
        },
        "inputs": [],
        "outputs": [checkpoint.to_dict()],
        "model": model_metadata,
        "data": {**dict(data_metadata or {}), "preference_diagnostics": diagnostics},
        "parameters": parameters,
        "hardware": provenance.collect_hardware(),
        "environment": {
            **provenance.collect_environment(),
            "metrics": metrics,
            "event_head": {
                key: start_event[key] for key in ("sequence", "last_event_hash", "log_sha256")
            },
        },
        "parents": [],
        "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(model_metadata)),
        "error": None,
    }
    validate_run_card(card)
    atomic_write_json(run_staging / "run_card.json", card)
    os.replace(staging, final)
    os.replace(run_staging, run_final)
    events.append(
        "run_finished",
        run_id,
        {"status": status, "run_card_sha256": sha256_file(run_final / "run_card.json")},
    )
    if status == "succeeded":
        CandidateRegistry(state_root).rebuild(policy)
    return {
        "status": status,
        "run_id": run_id,
        "method": method,
        "checkpoint": str(final),
        "metrics": metrics,
    }


def _manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = [item for item in manifest["shards"] if item["role"] == "preference_shard"]
    if not refs:
        raise PreferenceError("verified manifest has no preference shard")
    rows: list[dict[str, Any]] = []
    for ref in refs:
        path = Path(ref["path"])
        path = path if path.is_absolute() else ROOT / path
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def run_cli(args) -> tuple[dict[str, Any], int]:
    import torch

    if not torch.cuda.is_available():
        raise PreferenceError("real preference training requires CUDA")
    policy = load_policy()
    config = config_module.load_experiment(ROOT / args.config)
    manifest = verify_data_manifest(OUTPUTS / "data", policy)
    method = args.method or config.document["preference"]["method"]
    result = train_preference_adapter(
        method=method,
        model_source=policy.seed_model["repo_id"],
        revision=policy.seed_model["revision"],
        rows=_manifest_rows(manifest),
        output_root=OUTPUTS / "checkpoints",
        policy=policy,
        training=config.document["training"],
        preference=config.document["preference"],
        target_modules=config.document["training"]["target_modules"],
        device="cuda:0",
        qlora=config.document["training"]["method"] == "qlora",
        allow_checkpoints=args.allow_checkpoints,
        budget_minutes=args.budget_minutes or config.document["resources"]["budget_minutes"],
        data_metadata=manifest,
    )
    return result, 0 if result["status"] == "succeeded" else 1
