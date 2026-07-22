"""Real SFT/QLoRA and CPT adapter training on the pinned TRL stack."""

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
    sha256_directory,
    sha256_file,
    validate_run_card,
)
from .data_pipeline import verify_data_manifest
from .policy import Policy, load_policy
from .resolver import CandidateRegistry, OUTPUTS, load_tokenizer

from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[2]


class TrainingError(RuntimeError):
    """Training preflight, execution, deadline, or checkpoint validation failed."""


def validate_target_modules(model, targets: list[str]) -> None:
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(targets) - module_names)
    if missing:
        raise TrainingError(f"LoRA target modules do not exist in model: {missing}")


def validate_tokenizer(tokenizer, context_length: int, model) -> None:
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise TrainingError("tokenizer requires explicit EOS and pad tokens")
    maximum = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", None
    )
    if maximum is not None and context_length > maximum:
        raise TrainingError(f"context length {context_length} exceeds model limit {maximum}")
    if not tokenizer.chat_template:
        raise TrainingError("tokenizer chat template is missing")


def load_manifest_rows(manifest: Mapping[str, Any], kind: str, *, root: Path = ROOT):
    from datasets import Dataset

    role = f"{kind}_shard"
    refs = [item for item in manifest["shards"] if item["role"] == role]
    if not refs:
        raise TrainingError(f"data manifest contains no {role}")
    rows: list[dict[str, Any]] = []
    for ref in refs:
        path = Path(ref["path"])
        path = path if path.is_absolute() else root / path
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise TrainingError(f"{role} is empty")
    if kind == "sft":
        return Dataset.from_list([{"messages": row["messages"]} for row in rows])
    return Dataset.from_list([{"text": row["text"]} for row in rows])


def create_model_and_tokenizer(
    model_source: str,
    *,
    revision: str | None,
    qlora: bool,
    gradient_checkpointing: bool,
    policy: Policy,
    device: str,
):
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    local = Path(model_source).is_absolute()
    tokenizer = load_tokenizer(model_source, revision=revision, local_files_only=local)
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local,
        "dtype": torch.float32 if device == "cpu" else torch.bfloat16,
    }
    if qlora:
        if device == "cpu" or not torch.cuda.is_available():
            raise TrainingError("QLoRA requires CUDA and cannot fall back to CPU or LoRA")
        qlora_policy = policy.document["training"]["qlora"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_policy["quant_type"],
            bnb_4bit_use_double_quant=qlora_policy["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    else:
        model.to(device)
    return model, tokenizer


def collect_model_metadata(
    model_source: str,
    revision: str | None,
    model,
    policy: Policy,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    if model_source == policy.seed_model["repo_id"] and revision != policy.seed_model["revision"]:
        raise TrainingError("protected seed training requires its exact policy revision")
    local = Path(model_source).is_absolute()
    source = (
        Path(model_source)
        if local
        else Path(
            snapshot_download(
                model_source,
                revision=revision,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                ],
            )
        )
    )
    required = {
        "model_config_sha256": source / "config.json",
        "tokenizer_sha256": source / "tokenizer.json",
        "tokenizer_config_sha256": source / "tokenizer_config.json",
        "chat_template_sha256": source / "chat_template.jinja",
    }
    if any(not path.is_file() for path in required.values()):
        raise TrainingError("model source lacks config, tokenizer, or chat-template files")
    protected_source = (
        model_source == policy.seed_model["repo_id"] and revision == policy.seed_model["revision"]
    )
    metadata = {
        "base_model": {
            "repo_id": model_source,
            "revision": (
                policy.seed_model["revision"]
                if protected_source
                else sha256_directory(source)
                if local
                else str(revision)
            ),
        },
        **{key: sha256_file(path) for key, path in required.items()},
        "architecture": model.__class__.__name__,
    }
    if protected_source:
        expected = policy.seed_model
        if (
            metadata["base_model"]
            != {"repo_id": expected["repo_id"], "revision": expected["revision"]}
            or metadata["model_config_sha256"] != expected["model_config_sha256"]
            or metadata["tokenizer_sha256"] != expected["tokenizer_sha256"]
            or metadata["tokenizer_config_sha256"] != expected["tokenizer_config_sha256"]
            or metadata["chat_template_sha256"] != expected["chat_template_sha256"]
            or metadata["architecture"] != expected["architecture"]
        ):
            raise TrainingError("loaded model fingerprints differ from protected seed policy")
    return metadata


class DeadlineCallback(TrainerCallback):
    """Transformers callback that stops cleanly at the next step boundary."""

    def __init__(self, deadline: float):
        self.deadline = deadline
        self.exhausted = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline:
            control.should_training_stop = True
            self.exhausted = True
        return control


def _checkpoint_smoke(checkpoint: Path, model_source: str, revision: str | None) -> None:
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    config_path = checkpoint / "adapter_config.json"
    weights = checkpoint / "adapter_model.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise TrainingError("adapter checkpoint lacks config or safetensors weights")
    with safe_open(weights, framework="pt") as handle:
        if not list(handle.keys()):
            raise TrainingError("adapter safetensors contains no parameters")
    local = Path(model_source).is_absolute()
    base = AutoModelForCausalLM.from_pretrained(
        model_source, revision=revision, local_files_only=local, dtype=torch.float32
    )
    adapter = PeftModel.from_pretrained(base, checkpoint)
    tokenizer = load_tokenizer(model_source, revision=revision, local_files_only=local)
    encoded = tokenizer("Hallo", return_tensors="pt")
    with torch.inference_mode():
        adapter(**encoded)


def train_adapter(
    *,
    kind: str,
    model_source: str,
    revision: str | None,
    dataset,
    output_root: Path,
    policy: Policy,
    experiment: Mapping[str, Any],
    target_modules: list[str],
    device: str,
    qlora: bool,
    allow_checkpoints: bool,
    budget_minutes: float,
    repository_root: Path = ROOT,
    input_artifacts: list[Mapping[str, Any]] | None = None,
    parent_run_ids: list[str] | None = None,
    lineage: Mapping[str, Any] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not allow_checkpoints:
        raise TrainingError("checkpoint writes require explicit permission")
    if kind not in {"sft", "cpt"}:
        raise TrainingError(f"unsupported trainer kind {kind}")
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    run_id = new_run_id(f"train-{kind}")
    staging = output_root / ".staging" / run_id
    final_checkpoint = output_root / run_id
    state_root = output_root.parent
    run_staging = state_root / "runs/.staging" / run_id
    run_final = state_root / "runs" / run_id
    started = time.monotonic()
    deadline = started + budget_minutes * 60
    model, tokenizer = create_model_and_tokenizer(
        model_source,
        revision=revision,
        qlora=qlora,
        gradient_checkpointing=experiment["gradient_checkpointing"],
        policy=policy,
        device=device,
    )
    validate_tokenizer(tokenizer, experiment["context_length"], model)
    model_metadata = collect_model_metadata(model_source, revision, model, policy)
    validate_target_modules(model, target_modules)
    if (
        kind == "cpt"
        and experiment["learning_rate"] > policy.document["training"]["cpt_max_learning_rate"]
    ):
        raise TrainingError("CPT learning rate exceeds protected maximum")
    if experiment["assistant_only_loss"]:
        try:
            probe = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        except Exception as exc:
            raise TrainingError("assistant-only loss requires a template generation mask") from exc
        if not any(probe.get("assistant_masks", [])):
            raise TrainingError("assistant-only loss mask is empty")
    events = EventLog(state_root)
    start_event = events.append(
        "run_started", run_id, {"run_type": "train_sft" if kind == "sft" else "train_cpt"}
    )
    staging.mkdir(parents=True)
    callback = DeadlineCallback(deadline)
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=experiment["lora_r"],
        lora_alpha=experiment["lora_alpha"],
        lora_dropout=experiment["lora_dropout"],
        target_modules=target_modules,
        bias="none",
    )
    args = SFTConfig(
        output_dir=str(staging / "trainer"),
        per_device_train_batch_size=experiment["per_device_batch_size"],
        gradient_accumulation_steps=experiment["gradient_accumulation_steps"],
        learning_rate=experiment["learning_rate"],
        num_train_epochs=experiment["num_train_epochs"],
        max_steps=experiment["max_steps"],
        warmup_ratio=experiment["warmup_ratio"],
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=experiment["seed"],
        data_seed=experiment["seed"],
        use_cpu=device == "cpu",
        bf16=device != "cpu",
        gradient_checkpointing=experiment["gradient_checkpointing"],
        max_length=experiment["context_length"],
        packing=experiment["packing"],
        dataset_text_field="text" if kind == "cpt" else "text",
        assistant_only_loss=experiment["assistant_only_loss"] if kind == "sft" else False,
        completion_only_loss=False if kind == "cpt" else None,
        eos_token=tokenizer.eos_token,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[callback],
    )
    result = trainer.train()
    trainer.save_model(staging)
    tokenizer.save_pretrained(staging)
    if not (staging / "chat_template.jinja").exists() and tokenizer.chat_template:
        (staging / "chat_template.jinja").write_text(tokenizer.chat_template)
    _checkpoint_smoke(staging, model_source, revision)
    status = "budget_exhausted" if callback.exhausted else "succeeded"
    metrics = {
        "train_loss": float(result.training_loss),
        "steps_completed": int(trainer.state.global_step),
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "examples_seen": int(trainer.state.global_step)
        * experiment["per_device_batch_size"]
        * experiment["gradient_accumulation_steps"],
        "tokens_seen": int(trainer.state.num_input_tokens_seen or 0),
        "tokens_per_second": float(result.metrics.get("train_tokens_per_second", 0.0)),
        "wall_clock_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device != "cpu" else 0,
        "trainable_parameters": sum(
            value.numel() for value in trainer.model.parameters() if value.requires_grad
        ),
        "total_parameters": sum(value.numel() for value in trainer.model.parameters()),
        "effective_batch_size": experiment["per_device_batch_size"]
        * experiment["gradient_accumulation_steps"],
        "learning_rate": experiment["learning_rate"],
        "stop_reason": status,
    }
    checkpoint_ref = ArtifactRef.from_path(
        staging,
        role="adapter_checkpoint",
        media_type="application/vnd.boldt.peft-adapter",
    )
    try:
        stored_path = final_checkpoint.relative_to(repository_root).as_posix()
    except ValueError:
        stored_path = str(final_checkpoint)
    checkpoint_ref = ArtifactRef(
        stored_path,
        checkpoint_ref.kind,
        checkpoint_ref.role,
        checkpoint_ref.sha256,
        checkpoint_ref.size_bytes,
        checkpoint_ref.media_type,
    )
    run_staging.mkdir(parents=True)
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    card = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": "train_sft" if kind == "sft" else "train_cpt",
        "mode": "real",
        "status": status,
        "started_at": start_event["event"]["timestamp"],
        "finished_at": now,
        "duration_seconds": metrics["wall_clock_seconds"],
        "command": ["python", "-m", "boldt_posttrain.cli", "train", kind, "--real"],
        "git": provenance.collect_git("HEAD", root=repository_root),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "in-memory",
            "sha256": sha256_bytes(canonical_json_bytes(experiment)),
            "resolved_sha256": sha256_bytes(canonical_json_bytes(experiment)),
        },
        "inputs": [dict(item) for item in (input_artifacts or [])],
        "outputs": [checkpoint_ref.to_dict()],
        "model": model_metadata,
        "data": dict(data_metadata or {}),
        "parameters": {
            **dict(experiment),
            **({"lineage": dict(lineage)} if lineage is not None else {}),
        },
        "hardware": provenance.collect_hardware(),
        "environment": {
            **provenance.collect_environment(),
            "metrics": metrics,
            "event_head": {
                key: start_event[key] for key in ("sequence", "last_event_hash", "log_sha256")
            },
        },
        "parents": list(parent_run_ids or []),
        "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(model_metadata)),
        "error": None,
    }
    validate_run_card(card)
    atomic_write_json(run_staging / "run_card.json", card)
    os.replace(staging, final_checkpoint)
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
        "checkpoint": str(final_checkpoint),
        "metrics": metrics,
    }


def doctor(*, mode: str = "all", real: bool = False) -> dict[str, Any]:
    import importlib.metadata
    import shutil
    import torch

    result: dict[str, Any] = {
        "status": "succeeded",
        "mode": mode,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "bitsandbytes")
        },
        "hardware": provenance.collect_hardware(),
        "disk": {
            "free_bytes": shutil.disk_usage(ROOT).free,
            "total_bytes": shutil.disk_usage(ROOT).total,
        },
        "mergekit_executable": shutil.which("mergekit-yaml"),
    }
    if real:
        if not result["cuda_available"]:
            raise TrainingError("real doctor requires an available CUDA GPU")
        from .evaluation import validate_lm_eval_tasks
        from .resolver import resolve_hub_model

        policy = load_policy()
        resolved = resolve_hub_model(policy.seed_model["repo_id"], policy)
        result["seed_model"] = resolved.to_dict()
        result["eval_tasks"] = validate_lm_eval_tasks(policy)
        if result["mergekit_executable"] is None:
            raise TrainingError("mergekit-yaml executable is unavailable")
    return result


def run_cli(args) -> tuple[dict[str, Any], int]:
    if args.command == "doctor":
        result = doctor(mode=args.mode, real=args.real)
        return result, 0
    import torch

    if not torch.cuda.is_available():
        raise TrainingError("real training requires CUDA and cannot fall back to CPU")
    policy = load_policy()
    config = config_module.load_experiment(ROOT / args.config)
    manifest = verify_data_manifest(OUTPUTS / "data", policy)
    kind = args.train_command
    dataset = load_manifest_rows(manifest, kind)
    training = config.document["training"]
    result = train_adapter(
        kind=kind,
        model_source=policy.seed_model["repo_id"],
        revision=policy.seed_model["revision"],
        dataset=dataset,
        output_root=OUTPUTS / "checkpoints",
        policy=policy,
        experiment=training,
        target_modules=training["target_modules"],
        device="cuda:0",
        qlora=training["method"] == "qlora",
        allow_checkpoints=args.allow_checkpoints,
        budget_minutes=args.budget_minutes or config.document["resources"]["budget_minutes"],
        data_metadata=manifest,
    )
    return result, 0 if result["status"] == "succeeded" else 1
