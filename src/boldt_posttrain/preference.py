"""Conversational preference data and DPO/KTO/ORPO training helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence


METHODS = {"dpo", "kto", "orpo"}


class PreferenceResult:
    def __init__(self, output: Any, trainer: Any, stop_reason: Optional[str]):
        self.output = output
        self.trainer = trainer
        self.stop_reason = stop_reason

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)


def _messages(value: Any, role: str) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{role} conversation must be a non-empty list")
    from .data_pipeline import _message

    output = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("invalid conversational preference message")
        output.append(_message(message, default_role=role))
    return output


def normalize_preference_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the single protected conversational representation used by all trainers."""
    prompt = _messages(row.get("prompt"), "user")
    chosen = _messages(row.get("chosen"), "assistant")
    rejected = _messages(row.get("rejected"), "assistant")
    if prompt[-1]["role"] != "user":
        raise ValueError("preference prompt must end with a user message")
    if chosen[0]["role"] != "assistant" or rejected[0]["role"] != "assistant":
        raise ValueError("chosen and rejected must begin with assistant messages")
    result = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if row.get("tools") is not None:
        if not isinstance(row["tools"], list):
            raise ValueError("preference tools must be a list")
        result["tools"] = row["tools"]
    return result


def validate_preference_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_completion_length: int,
    context_length: int,
    length_ratio_max: float,
) -> Dict[str, Any]:
    from .data_pipeline import _template_encoding, distribution

    prompts: List[int] = []
    chosen_lengths: List[int] = []
    rejected_lengths: List[int] = []
    ratios: List[float] = []
    for index, raw in enumerate(rows):
        row = normalize_preference_row(raw)
        if row["prompt"][-1]["role"] != "user":
            raise ValueError(f"preference row {index} prompt must end in user")
        if not any(message["role"] == "assistant" for message in row["chosen"]):
            raise ValueError(f"preference row {index} chosen has no assistant completion")
        if not any(message["role"] == "assistant" for message in row["rejected"]):
            raise ValueError(f"preference row {index} rejected has no assistant completion")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"preference row {index} chosen and rejected are identical")
        prompt_ids, _ = _template_encoding(
            tokenizer,
            row["prompt"],
            tools=row.get("tools"),
            add_generation_prompt=True,
        )
        chosen_ids, chosen_mask = _template_encoding(tokenizer, row["chosen"], assistant_mask=True)
        rejected_ids, rejected_mask = _template_encoding(
            tokenizer, row["rejected"], assistant_mask=True
        )
        if not chosen_mask or sum(chosen_mask) == 0:
            raise ValueError(f"preference row {index} chosen has no assistant tokens")
        if not rejected_mask or sum(rejected_mask) == 0:
            raise ValueError(f"preference row {index} rejected has no assistant tokens")
        prompt_length = len(prompt_ids)
        chosen_length = len(chosen_ids)
        rejected_length = len(rejected_ids)
        ratio = max(chosen_length, rejected_length) / max(1, min(chosen_length, rejected_length))
        if prompt_length > max_prompt_length:
            raise ValueError(f"preference row {index} prompt would be truncated")
        if max(chosen_length, rejected_length) > max_completion_length:
            raise ValueError(f"preference row {index} completion would be truncated")
        if prompt_length + max(chosen_length, rejected_length) > context_length:
            raise ValueError(f"preference row {index} exceeds the model context")
        if ratio > length_ratio_max:
            raise ValueError(f"preference row {index} completion length ratio exceeds limit")
        prompts.append(prompt_length)
        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        ratios.append(ratio)
    return {
        "prompt_tokens": distribution(prompts),
        "chosen_completion_tokens": distribution(chosen_lengths),
        "rejected_completion_tokens": distribution(rejected_lengths),
        "completion_length_ratio": distribution(ratios),
        "rows": len(rows),
    }


def iter_preference_jsonl(paths: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    yield normalize_preference_row(row)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid preference row at {path}:{line_number}: {exc}"
                    ) from exc


def render_probe(tokenizer: Any, probe: Mapping[str, Any]) -> Dict[str, str]:
    row = normalize_preference_row(probe)
    return {
        key: tokenizer.apply_chat_template(
            row["prompt"] + row[key], tokenize=False, add_generation_prompt=False
        )
        for key in ("chosen", "rejected")
    }


def run_response_suppression_probes(
    model: Any,
    tokenizer: Any,
    probes: Sequence[Mapping[str, Any]],
    *,
    device: str,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Run a protected probe set; callers must not pass arbitrary training prefixes."""
    if not probes:
        raise ValueError("the protected response-suppression probe set is empty")
    import torch

    empty = refusals = 0
    from .evaluation import is_refusal

    for probe in probes:
        prompt = normalize_preference_row(probe)["prompt"]
        rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        empty += int(not output)
        refusals += int(is_refusal(output))
    return {"empty_output_rate": empty / len(probes), "refusal_rate": refusals / len(probes)}


class DeadlineCallback:
    """Transformers callback that terminates only at a trainer step boundary."""

    def __init__(self, deadline: float, reserve_seconds: float = 0.0):
        self.deadline = deadline
        self.stop_at = deadline - reserve_seconds
        self.stop_reason: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        # Keep the core package importable without transformers while satisfying
        # the complete TrainerCallback event surface when the train extra is present.
        if name.startswith("on_"):
            return lambda _args, _state, control, **_kwargs: control
        raise AttributeError(name)

    def on_step_end(self, _args: Any, _state: Any, control: Any, **_kwargs: Any) -> Any:
        if time.monotonic() >= self.stop_at:
            control.should_training_stop = True
            self.stop_reason = "budget_limit"
        return control


def bind_training_arguments_device(args: Any, device: str) -> Any:
    """Bind the pinned Transformers arguments to one explicit device, never DataParallel."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("device binding requires torch") from exc
    target = torch.device(device)
    # Force initialization before replacing the cached device selected by TrainingArguments.
    _ = args.device
    if target.type == "cuda":
        torch.cuda.set_device(target)
        args._n_gpu = 1
    elif target.type == "cpu":
        args._n_gpu = 0
    else:
        raise ValueError(f"unsupported training device: {device}")
    if getattr(args, "distributed_state", None) is not None:
        args.distributed_state.device = target
    args.__dict__["_setup_devices"] = target
    if args.device != target:
        raise RuntimeError(f"trainer selected {args.device}, expected {target}")
    return args


def train_preference(
    *,
    method: str,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    eval_dataset: Any = None,
    output_dir: Path,
    training_args: Mapping[str, Any],
    preference_config: Optional[Mapping[str, Any]] = None,
    peft_config: Any = None,
    deadline: Optional[float] = None,
    reserve_seconds: float = 0.0,
    device: Optional[str] = None,
) -> Any:
    """Train one conversational preference method using the installed, single TRL API."""
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"preference method must be one of {sorted(METHODS)}")
    try:
        from transformers import TrainingArguments
        from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer, ORPOConfig, ORPOTrainer
    except ImportError as exc:
        raise RuntimeError("preference training requires the train extra") from exc
    trainer_cls, config_cls = {
        "dpo": (DPOTrainer, DPOConfig),
        "kto": (KTOTrainer, KTOConfig),
        "orpo": (ORPOTrainer, ORPOConfig),
    }[method]
    if method != "dpo" and float((preference_config or {}).get("sft_loss_weight", 0.0)) > 0:
        raise ValueError("preference.sft_loss_weight is supported only for DPO")

    def to_kto(value: Any) -> Any:
        if value is None:
            return None
        columns = getattr(value, "column_names", None)

        def preference_to_kto(batch: Mapping[str, Sequence[Any]]) -> Dict[str, List[Any]]:
            prompts, completions, labels = [], [], []
            for prompt, chosen, rejected in zip(
                batch["prompt"], batch["chosen"], batch["rejected"]
            ):
                prompts.extend([prompt, prompt])
                completions.extend([chosen, rejected])
                labels.extend([True, False])
            return {"prompt": prompts, "completion": completions, "label": labels}

        return value.map(
            preference_to_kto,
            batched=True,
            remove_columns=columns if columns else ["chosen", "rejected"],
        )

    if method == "kto":
        dataset = to_kto(dataset)
        eval_dataset = to_kto(eval_dataset)
    # Config classes derive from TrainingArguments on the supported TRL release.
    config_kwargs = dict(training_args)
    config_kwargs["output_dir"] = str(output_dir)
    if method == "dpo" and preference_config is not None:
        primary = str(preference_config.get("loss_type", "sigmoid"))
        sft_weight = float(preference_config.get("sft_loss_weight", 0.0))
        config_kwargs["beta"] = float(preference_config.get("beta", 0.1))
        config_kwargs["loss_type"] = [primary, "sft"] if sft_weight > 0 else [primary]
        config_kwargs["loss_weights"] = [1.0, sft_weight] if sft_weight > 0 else None
    if preference_config is not None:
        config_kwargs["max_prompt_length"] = int(preference_config.get("max_prompt_length", 4096))
        if method == "dpo":
            config_kwargs["max_completion_length"] = int(
                preference_config.get("max_completion_length", 2048)
            )
    args = config_cls(**config_kwargs)
    if not isinstance(args, TrainingArguments):
        raise RuntimeError("installed TRL preference config is incompatible")
    if device is not None:
        bind_training_arguments_device(args, device)
    deadline_callback = (
        DeadlineCallback(deadline, reserve_seconds=reserve_seconds)
        if deadline is not None
        else None
    )
    callbacks = [deadline_callback] if deadline_callback is not None else []
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    return PreferenceResult(
        result,
        trainer,
        deadline_callback.stop_reason if deadline_callback else None,
    )
