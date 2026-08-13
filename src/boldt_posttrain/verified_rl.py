"""One GRPO operator for mathematically verifiable conversational tasks."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .preference import DeadlineCallback
from .training import (
    _checkpoint_smoke_subprocess,
    align_quantized_output_head_input,
    evaluation_interval,
    load_tokenizer,
    load_trainable_adapter,
    make_peft_config,
    validate_liger,
)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], Mapping):
        return str(completion[-1].get("content", ""))
    raise ValueError("GRPO completion must be text or a conversational assistant completion")


def parse_gold_solution(solution: str) -> Any:
    try:
        from math_verify import parse
    except ImportError as exc:
        raise RuntimeError("verified-math GRPO requires the rl extra") from exc
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("verified-math gold solution must be non-empty text")
    parsed = parse(solution)
    if not parsed:
        raise ValueError(f"math-verify could not parse gold solution: {solution!r}")
    return parsed


def math_accuracy_reward(
    completions: Sequence[Any], solution: Sequence[str], **_kwargs: Any
) -> list[float]:
    """Exact Math-Verify accuracy; malformed predictions receive zero, never partial credit."""
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError("verified-math GRPO requires the rl extra") from exc
    rewards = []
    for completion, gold_text in zip(completions, solution):
        gold = parse_gold_solution(gold_text)
        prediction = parse(_completion_text(completion))
        rewards.append(float(bool(prediction) and verify(gold, prediction)))
    return rewards


def validate_verified_math_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_length: int,
) -> Dict[str, Any]:
    from .data_pipeline import _template_encoding, distribution

    prompt_lengths = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt or prompt[-1].get("role") != "user":
            raise ValueError(f"verified_math row {row.get('content_id', index)} has invalid prompt")
        try:
            parse_gold_solution(row.get("solution"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"verified_math row {row.get('content_id', index)} has unparseable gold: {exc}"
            ) from exc
        ids, _ = _template_encoding(tokenizer, prompt, add_generation_prompt=True)
        if len(ids) > max_prompt_length:
            raise ValueError(
                f"verified_math row {row.get('content_id', index)} prompt would be truncated"
            )
        prompt_lengths.append(len(ids))
    return {"rows": len(rows), "prompt_tokens": distribution(prompt_lengths)}


def make_grpo_config(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    device: str,
    has_validation: bool,
) -> Any:
    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise RuntimeError("verified-math GRPO requires the train extra") from exc
    training = config["training"]
    verified = config["verified_rl"]
    generations = int(verified["num_generations"])
    batch_size = int(verified.get("batch_size", generations))
    if generations < 2:
        raise ValueError("verified_rl.num_generations must be at least two")
    if batch_size % generations:
        raise ValueError("verified_rl.batch_size must be divisible by num_generations")
    if int(verified["max_prompt_length"]) + int(verified["max_completion_length"]) > int(
        training["context_length"]
    ):
        raise ValueError("verified RL prompt and completion limits exceed model context")
    steps = int(training["max_steps"])
    interval = evaluation_interval(steps)
    metric = "eval_rewards/math_accuracy_reward/mean"
    return GRPOConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        num_generations=generations,
        max_prompt_length=int(verified["max_prompt_length"]),
        max_completion_length=int(verified["max_completion_length"]),
        temperature=float(verified["temperature"]),
        beta=float(verified["beta"]),
        use_vllm=False,
        use_liger_kernel=bool(training.get("use_liger_kernel", False)),
        bf16=device.startswith("cuda:"),
        use_cpu=not device.startswith("cuda:"),
        report_to="none",
        logging_steps=1,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=interval if has_validation else None,
        save_strategy="steps" if has_validation else "no",
        save_steps=interval,
        save_total_limit=1 if has_validation else None,
        load_best_model_at_end=has_validation,
        metric_for_best_model=metric if has_validation else None,
        greater_is_better=True if has_validation else None,
        remove_unused_columns=False,
        mask_truncated_completions=True,
        seed=int(training["seed"]),
    )


def train_verified_grpo(
    *,
    model_ref: str,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: Path,
    config: Mapping[str, Any],
    device: str,
    deadline: float,
    parent_adapter: Optional[Path] = None,
) -> Dict[str, Any]:
    verified = config.get("verified_rl", {})
    if verified.get("enabled") is not True:
        raise ValueError("verified-math GRPO is disabled by verified_rl.enabled=false")
    if verified.get("reward_profile") != "math_accuracy":
        raise ValueError("the only supported GRPO reward profile is math_accuracy")
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("verified-math GRPO requires non-empty train and validation datasets")
    training = config["training"]
    if training.get("method") not in {"lora", "qlora"}:
        raise ValueError("verified-math GRPO supports only explicit lora or qlora training")
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )
        from trl import GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("verified-math GRPO requires the train and rl extras") from exc
    tokenizer = load_tokenizer(model_ref, revision=training.get("revision"))
    diagnostics = validate_verified_math_rows(
        list(train_dataset) + list(eval_dataset),
        tokenizer,
        max_prompt_length=int(verified["max_prompt_length"]),
    )
    model_config = AutoConfig.from_pretrained(model_ref, revision=training.get("revision"))
    validate_liger(
        bool(training.get("use_liger_kernel", False)),
        model_type=str(model_config.model_type),
    )
    remaining = deadline - time.monotonic()
    reserve_seconds = min(300.0, max(15.0, remaining * 0.10))
    if remaining <= reserve_seconds:
        raise RuntimeError("GRPO budget is too small for validation, save, and reload reserve")
    trainer_dir = Path(output_dir) / "trainer"
    adapter_dir = Path(output_dir) / "adapter"
    adapter_staging = Path(output_dir) / ".adapter-staging"
    args = make_grpo_config(config, trainer_dir, device=device, has_validation=True)
    quantization = None
    if training.get("method") == "qlora":
        if not device.startswith("cuda:"):
            raise ValueError("QLoRA requires an explicit CUDA device")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    compute_dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    device_map: Any = (
        {"": int(device.split(":", 1)[1])} if device.startswith("cuda:") else {"": "cpu"}
    )
    model_input: Any = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision=training.get("revision"),
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map=device_map,
    )
    peft_config = make_peft_config(training)
    if parent_adapter is not None:
        model_input = load_trainable_adapter(
            model_input,
            parent_adapter,
        )
        peft_config = None
    callback = DeadlineCallback(deadline, reserve_seconds=reserve_seconds)

    class _VerifiedMathGRPOTrainer(GRPOTrainer):
        def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
            # TRL 0.23 logs reward metrics but does not include them in evaluate()'s return
            # mapping. Mutating the mapping lets Transformers select the best checkpoint by
            # the exact same validation accuracy without implementing another eval loop.
            if not self.model.training:
                logs.update(
                    {
                        f"eval_{key}": sum(values) / len(values)
                        for key, values in self._metrics["eval"].items()
                        if values
                    }
                )
            super().log(logs, start_time)

    trainer = _VerifiedMathGRPOTrainer(
        model=model_input,
        reward_funcs=math_accuracy_reward,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
        peft_config=peft_config,
    )
    align_quantized_output_head_input(trainer.model)
    started = time.monotonic()
    output = trainer.train()
    elapsed = time.monotonic() - started
    history = list(trainer.state.log_history)
    eval_rows = [row for row in history if "eval_rewards/math_accuracy_reward/mean" in row]
    if not eval_rows:
        evaluated = trainer.evaluate()
        if "eval_rewards/math_accuracy_reward/mean" in evaluated:
            eval_rows.append(dict(evaluated) | {"step": trainer.state.global_step})
    if not eval_rows:
        raise RuntimeError("GRPO validation produced no verified-math accuracy")
    trainer.save_model(str(adapter_staging))
    tokenizer.save_pretrained(str(adapter_staging))
    _checkpoint_smoke_subprocess(
        adapter_staging,
        base_model=model_ref,
        revision=training.get("revision"),
    )
    adapter_staging.replace(adapter_dir)
    shutil.rmtree(trainer_dir, ignore_errors=True)
    latest = history[-1] if history else {}

    def last_metric(name: str) -> Any:
        return next((row[name] for row in reversed(history) if name in row), None)

    best = max(eval_rows, key=lambda row: row["eval_rewards/math_accuracy_reward/mean"])
    tokens = float(last_metric("num_tokens") or output.metrics.get("num_input_tokens_seen", 0))
    metrics = {
        **dict(output.metrics),
        "status": "succeeded",
        "steps": int(trainer.state.global_step),
        "stop_reason": callback.stop_reason
        or ("max_steps" if trainer.state.global_step >= args.max_steps else "epochs_completed"),
        "validation_examples": len(eval_dataset),
        "validation_evaluations": len(eval_rows),
        "best_validation_accuracy": float(best["eval_rewards/math_accuracy_reward/mean"]),
        "best_step": int(best.get("step", 0)),
        "reward_mean": last_metric("reward"),
        "reward_std": last_metric("reward_std"),
        "frac_reward_zero_std": last_metric("frac_reward_zero_std"),
        "completion_clipping_fraction": last_metric("completions/clipped_ratio"),
        "mean_completion_length": last_metric("completions/mean_length"),
        "tokens_seen": tokens,
        "tokens_per_second": tokens / elapsed if elapsed else 0.0,
        "gpu_seconds": elapsed,
        "peak_vram": (
            int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda:") else 0
        ),
        "effective_options": {
            "reward_profile": "math_accuracy",
            "num_generations": args.num_generations,
            "use_liger_kernel": args.use_liger_kernel,
            "use_rslora": bool(training.get("use_rslora", False)),
        },
        "verified_math_diagnostics": diagnostics,
        "last_log": {
            key: latest[key]
            for key in (
                "reward",
                "reward_std",
                "frac_reward_zero_std",
                "completions/clipped_ratio",
                "completions/mean_length",
            )
            if key in latest
        },
    }
    return {"status": "succeeded", "adapter": str(adapter_dir), "metrics": metrics}
