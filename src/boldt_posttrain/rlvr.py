"""RLOO RLVR training over conversational, leakage-clean mechanically verifiable data."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

from .data_pipeline import FastTextLanguageIdentifier
from .preference import DeadlineCallback
from .rewards import REWARD_VERSION, total_reward
from .training import load_tokenizer, load_trainable_adapter, make_peft_config

TASK_TYPES = {"numeric", "json_schema", "exact", "ordered_terms", "language", "non_refusal"}


def validate_rlvr_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    required = {
        "prompt",
        "task_type",
        "ground_truth",
        "reward_version",
        "source",
        "license",
        "content_id",
    }
    missing = required - set(row)
    if missing:
        raise ValueError("RLVR row missing keys: " + ", ".join(sorted(missing)))
    prompt = row["prompt"]
    if not isinstance(prompt, list) or not prompt or prompt[-1].get("role") != "user":
        raise ValueError("RLVR prompt must be a conversational prompt ending in user")
    if row["task_type"] not in TASK_TYPES:
        raise ValueError(f"unsupported RLVR task type: {row['task_type']}")
    if row["reward_version"] != REWARD_VERSION:
        raise ValueError("unsupported RLVR reward version")
    if not isinstance(row["ground_truth"], dict) or not isinstance(row["source"], dict):
        raise ValueError("RLVR ground_truth and source must be objects")
    if row.get("leakage_clean") is not True or row.get("training_usable") is not True:
        raise ValueError("RLVR rows must be explicitly leakage-clean and trainable")
    return dict(row)


def iter_rlvr_rows(paths: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield validate_rlvr_row(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid RLVR row at {path}:{line_number}: {exc}") from exc


def make_reward_callable(
    *, weights: Mapping[str, float], clamp: Sequence[float], language_id: FastTextLanguageIdentifier
) -> Callable[..., list[float]]:
    """Adapt the fixed scalar registry to the installed RLOO batched callable contract."""

    def mechanical_reward(
        completions: Sequence[Any],
        task_type: Sequence[str],
        ground_truth: Sequence[Mapping[str, Any]],
        log_extra: Any = None,
        log_metric: Any = None,
        **_kwargs: Any,
    ) -> list[float]:
        values = []
        component_values: Dict[str, list[float]] = {}
        invalid = 0
        for completion, kind, truth in zip(completions, task_type, ground_truth):
            details = {}

            def log_component(name: str, value: Optional[float], reason: Optional[str]) -> None:
                details[name] = {"value": value, "reason": reason}
                if value is not None:
                    component_values.setdefault(name, []).append(value)

            value = total_reward(
                completion,
                task_type=kind,
                ground_truth=truth,
                weights=weights,
                clamp=clamp,
                language_id=language_id,
                log_component=log_component,
            )
            if not math.isfinite(value):
                raise ValueError("total reward is NaN or infinite")
            invalid += int(
                any(part["reason"] not in {None, "not_applicable"} for part in details.values())
            )
            values.append(value)
        if log_extra:
            log_extra("mechanical_reward", values)
        if log_metric:
            for name, parts in component_values.items():
                log_metric(f"reward_component/{name}", sum(parts) / len(parts))
            log_metric("fraction_zero_reward", sum(value == 0 for value in values) / len(values))
            log_metric("fraction_invalid", invalid / len(values))
        mechanical_reward.last_metrics = {
            **{
                f"reward_component/{name}": sum(parts) / len(parts)
                for name, parts in component_values.items()
            },
            "fraction_zero_reward": sum(value == 0 for value in values) / len(values),
            "fraction_invalid": invalid / len(values),
        }
        return values

    mechanical_reward.__name__ = "mechanical_reward"
    mechanical_reward.last_metrics = {}
    return mechanical_reward


def _checkpoint_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def checkpoint_smoke(adapter_path: Path, base_model: str, prompt: str, *, device: str) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("checkpoint smoke requires the train dependencies") from exc
    tokenizer = load_tokenizer(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**encoded)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("reloaded RLVR adapter produced non-finite logits")


def train_rlvr(
    *,
    model_ref: str,
    dataset: Any,
    output_dir: Path,
    config: Mapping[str, Any],
    policy: Mapping[str, Any],
    device: str,
    deadline: float,
    language_id: FastTextLanguageIdentifier,
    register_candidate: Optional[Callable[[Mapping[str, Any]], None]] = None,
    parent_adapter: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the one supported online-RL algorithm: RLOO with PEFT and callable rewards."""
    try:
        import torch
        from trl import RLOOConfig, RLOOTrainer
    except ImportError as exc:
        raise RuntimeError("RLVR training requires the locked train dependencies") from exc
    training = dict(config.get("training", {}))
    rlvr = dict(config.get("rlvr", {}))
    if rlvr.get("use_vllm", False):
        raise ValueError("RLVR vLLM is not supported in this version")
    if rlvr.get("use_liger_kernel", False):
        raise ValueError("RLOO has no enabled Liger path")
    args = RLOOConfig(
        output_dir=str(output_dir),
        max_steps=int(training["max_steps"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(rlvr.get("batch_size", 4)),
        gradient_accumulation_steps=int(rlvr.get("gradient_accumulation_steps", 1)),
        num_generations=int(rlvr.get("num_generations", 4)),
        max_completion_length=int(rlvr.get("max_completion_length", 256)),
        beta=float(rlvr.get("kl_coefficient", 0.05)),
        use_vllm=False,
        bf16=device.startswith("cuda:"),
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        use_liger_kernel=False,
    )
    reward = make_reward_callable(
        weights=policy["reward_weights"], clamp=policy["reward_clamp"], language_id=language_id
    )
    model_input: Any = model_ref
    peft_config = make_peft_config(training)
    if parent_adapter is not None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("RLVR continuation requires transformers") from exc
        model_input = load_trainable_adapter(
            AutoModelForCausalLM.from_pretrained(model_ref), parent_adapter
        )
        peft_config = None
    trainer = RLOOTrainer(
        model=model_input,
        args=args,
        reward_funcs=reward,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[DeadlineCallback(deadline)],
    )
    started = time.monotonic()
    result = trainer.train()
    elapsed = time.monotonic() - started
    if time.monotonic() >= deadline:
        return {
            "status": "budget_exhausted",
            "candidate_registered": False,
            "gpu_seconds": elapsed,
            **dict(result.metrics),
        }
    adapter = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter))
    checkpoint_smoke(adapter, model_ref, "Kurzer Test", device=device)
    history = trainer.state.log_history
    metrics = dict(result.metrics)
    tokens_seen = max(
        (float(row["num_tokens"]) for row in history if "num_tokens" in row), default=0.0
    )
    metrics.update(
        {
            "gpu_seconds": elapsed,
            "tokens_seen": tokens_seen,
            "tokens_per_second": tokens_seen / elapsed if elapsed > 0 else 0.0,
            "peak_vram": (
                int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda:") else 0
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in trainer.model.parameters()
                if parameter.requires_grad
            ),
            "checkpoint_bytes": _checkpoint_bytes(adapter),
            "generations_per_prompt": args.num_generations,
            "reward_mean": next(
                (row["reward"] for row in reversed(history) if "reward" in row), None
            ),
            "reward_std": next(
                (row["reward_std"] for row in reversed(history) if "reward_std" in row), None
            ),
            "kl": next((row["kl"] for row in reversed(history) if "kl" in row), None),
            "completion_length": next(
                (
                    row["completions/mean_length"]
                    for row in reversed(history)
                    if "completions/mean_length" in row
                ),
                None,
            ),
        }
    )
    metrics.update(reward.last_metrics)
    for row in history:
        for key, value in row.items():
            if key.startswith("reward_component/") or key in {
                "fraction_zero_reward",
                "fraction_invalid",
            }:
                metrics[key] = value
    final = {
        "status": "ok",
        "candidate_registered": register_candidate is not None,
        "adapter": str(adapter),
        "metrics": metrics,
    }
    if register_candidate is not None:
        register_candidate(final)
    return final
