"""Shared streaming LoRA/QLoRA training and explicit device capability checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import provenance as prov
from . import recipe

ROOT = Path(__file__).resolve().parents[2]

# Which optional preference loss / training knobs to surface in the dry-run plan, per kind.
_RUN_TYPE = {"specialist": "train_specialist", "preference": "train_preference", "cpt": "train_cpt"}


def load_tokenizer(model_ref: str, *, revision: Optional[str] = None) -> Any:
    """Load a tokenizer while adapting Transformers' legacy list-form special-token metadata."""
    try:
        from transformers import AutoTokenizer
        from transformers.models.auto.tokenization_auto import get_tokenizer_config
    except ImportError as exc:
        raise RuntimeError("tokenizer loading requires transformers") from exc
    tokenizer_config = get_tokenizer_config(model_ref, revision=revision)
    extra = tokenizer_config.get("extra_special_tokens")
    kwargs: Dict[str, Any] = {}
    if isinstance(extra, list):
        if not extra or not all(isinstance(token, str) and token for token in extra):
            raise ValueError("legacy tokenizer extra_special_tokens must be non-empty text")
        names: Dict[str, str] = {}
        for index, token in enumerate(extra):
            stem = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
            name = f"{stem}_token" if stem else f"extra_special_token_{index}"
            if name in names:
                name = f"{name}_{index}"
            names[name] = token
        kwargs["extra_special_tokens"] = names
    return AutoTokenizer.from_pretrained(model_ref, revision=revision, **kwargs)


def resolve_mix_plan(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return path
    candidate = ROOT / "outputs/posttrain/mix" / str(value) / "mix_plan.json"
    if not candidate.is_file():
        raise ValueError(f"mix plan run ID or path not found: {value}")
    document = json.loads(candidate.read_text(encoding="utf-8"))
    if document.get("run_id") != str(value):
        raise ValueError("mix plan run ID does not match its artifact")
    return candidate


def _training_plan(cfg: Dict[str, Any], kind: str, specialist: str) -> Dict[str, Any]:
    tr = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    plan = {
        "kind": kind,
        "specialist": specialist,
        "base_model": tr.get("base_model"),
        "method": tr.get("method"),
        "learning_rate": tr.get("learning_rate"),
        "num_train_epochs": tr.get("num_train_epochs"),
        "max_steps": tr.get("max_steps"),
        "context_length": tr.get("context_length"),
        "lora_r": tr.get("lora_r"),
        "lora_alpha": tr.get("lora_alpha"),
        "target_modules": tr.get("target_modules"),
    }
    if kind == "preference":
        plan["preference"] = cfg.get("preference", {})
    return plan


def _manifest_clean(
    data_dir: Path,
    *,
    expected_decontamination_hash: Optional[str] = None,
    expected_policy_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail-closed check that prepared data is trainable: manifest present + leakage verified clean."""
    manifest = data_dir / "manifest.json"
    leakage = data_dir / "leakage_report.json"
    if not manifest.exists():
        return {"clean": False, "reason": f"no data manifest at {manifest} — run /pt-data real"}
    if not leakage.exists():
        return {"clean": False, "reason": f"no leakage report at {leakage} — fails closed"}
    try:
        manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
        lk = json.loads(leakage.read_text(encoding="utf-8"))
        decontamination_path = data_dir / "decontamination.json"
        if expected_decontamination_hash is None and decontamination_path.exists():
            expected_decontamination_hash = json.loads(
                decontamination_path.read_text(encoding="utf-8")
            ).get("artifact_hash")
    except Exception:
        return {"clean": False, "reason": "leakage_report.json unparseable — fails closed"}
    status = str(lk.get("status", "")).lower()
    if status not in ("clean", "verified_clean", "ok"):
        return {"clean": False, "reason": f"leakage status '{status}' is not verified clean"}
    if (
        expected_decontamination_hash is not None
        and manifest_doc.get("decontamination_hash") != expected_decontamination_hash
    ):
        return {"clean": False, "reason": "data manifest decontamination hash is stale"}
    if expected_policy_hash is not None and manifest_doc.get("policy_hash") != expected_policy_hash:
        return {"clean": False, "reason": "data manifest policy hash is stale"}
    return {"clean": True, "reason": "manifest present and leakage verified clean"}


def validate_device(
    device: str, *, minimum_vram_gb: float, torch_module: Any = None, bitsandbytes_smoke: Any = None
) -> Dict[str, Any]:
    """Validate capabilities, never a specific GPU model or compute capability."""
    if not device.startswith("cuda:"):
        raise ValueError("training device must be explicit, for example cuda:0")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("device validation requires torch") from exc
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    try:
        index = int(device.split(":", 1)[1])
        properties = torch_module.cuda.get_device_properties(index)
    except (ValueError, IndexError, RuntimeError) as exc:
        raise RuntimeError(f"CUDA device is not visible: {device}") from exc
    vram_gb = properties.total_memory / (1024**3)
    if vram_gb < minimum_vram_gb:
        raise RuntimeError(f"{device} has {vram_gb:.2f} GiB; requires {minimum_vram_gb:.2f} GiB")
    bf16 = bool(torch_module.cuda.is_bf16_supported())
    if not bf16:
        raise RuntimeError(f"{device} does not support BF16")
    if bitsandbytes_smoke is None:

        def bitsandbytes_smoke() -> None:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise RuntimeError("bitsandbytes is not installed") from exc
            layer = bnb.nn.Linear4bit(16, 16, quant_type="nf4", compute_dtype=torch_module.bfloat16)
            layer.to(device)
            layer(torch_module.zeros((1, 16), device=device, dtype=torch_module.bfloat16))

    bitsandbytes_smoke()
    return {
        "device": device,
        "name": properties.name,
        "vram_gb": vram_gb,
        "bf16": bf16,
        "bitsandbytes_4bit": True,
    }


def make_peft_config(
    training: Dict[str, Any],
    *,
    modules_to_save: Optional[List[str]] = None,
    ensure_weight_tying: bool = False,
) -> Any:
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError("LoRA training requires peft") from exc
    recipe_name = str(training.get("lora_init", "default"))
    if recipe_name not in {"default", "pissa_niter_4"}:
        raise ValueError("lora_init must be 'default' or 'pissa_niter_4'")
    if recipe_name == "pissa_niter_4" and training.get("use_rslora"):
        raise ValueError("PiSSA and rsLoRA are separate supported recipes, not a combined recipe")
    return LoraConfig(
        r=int(training["lora_r"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training.get("lora_dropout", 0.0)),
        target_modules=list(training["target_modules"]),
        task_type="CAUSAL_LM",
        use_rslora=bool(training.get("use_rslora", False)),
        init_lora_weights=True if recipe_name == "default" else "pissa_niter_4",
        modules_to_save=list(modules_to_save) if modules_to_save else None,
        ensure_weight_tying=ensure_weight_tying,
    )


def evaluation_interval(planned_optimizer_steps: int) -> int:
    if planned_optimizer_steps <= 0:
        raise ValueError("planned optimizer steps must be positive")
    return max(1, min(100, planned_optimizer_steps // 10 or 1))


def validate_liger(enabled: bool, *, model_type: Optional[str] = None) -> None:
    if not enabled:
        return
    try:
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    except ImportError as exc:
        raise RuntimeError("use_liger_kernel=true requires the perf extra") from exc
    if model_type is not None and model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
        raise ValueError(f"Liger does not support model architecture {model_type!r}")


def align_quantized_output_head_input(model: Any) -> None:
    """Keep QLoRA generation compatible with PEFT's intentional FP32 output head."""
    if not (
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    ):
        return
    head = model.get_output_embeddings()
    if head is None or getattr(head, "_boldt_input_dtype_aligned", False):
        return

    def cast_input(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs or not hasattr(inputs[0], "to") or not hasattr(module, "weight"):
            return inputs
        return (inputs[0].to(dtype=module.weight.dtype), *inputs[1:])

    head.register_forward_pre_hook(cast_input)
    head._boldt_input_dtype_aligned = True


def _assistant_mask_metrics(
    rows: List[Dict[str, Any]], tokenizer: Any, context_length: int
) -> Dict[str, Any]:
    from .data_pipeline import _template_encoding

    supervised = total = empty = lost = 0
    for index, row in enumerate(rows):
        ids, mask = _template_encoding(
            tokenizer,
            row["messages"],
            tools=row.get("tools"),
            assistant_mask=True,
        )
        assert mask is not None
        previous_length = 0
        for message_index, message in enumerate(row["messages"]):
            prefix_ids, prefix_mask = _template_encoding(
                tokenizer,
                row["messages"][: message_index + 1],
                tools=row.get("tools"),
                assistant_mask=True,
            )
            assert prefix_mask is not None
            if message.get("role") != "assistant" and sum(prefix_mask[previous_length:]):
                raise ValueError(
                    f"SFT row {row.get('content_id', index)} marks non-assistant tokens as assistant"
                )
            previous_length = len(prefix_ids)
        count = sum(mask[:context_length])
        full_count = sum(mask)
        if full_count == 0:
            empty += 1
        if count == 0:
            lost += 1
            raise ValueError(
                f"SFT row {row.get('content_id', index)} has no assistant tokens after truncation"
            )
        supervised += count
        total += min(len(ids), context_length)
    return {
        "enabled": True,
        "supervised_token_fraction": supervised / total if total else 0.0,
        "examples_without_supervised_tokens": empty,
        "truncated_without_supervision": lost,
    }


def validate_assistant_supervision(
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    tokenizer: Any,
    *,
    context_length: int,
    manifest_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("assistant-only SFT requires a tokenizer chat template")
    metrics = _assistant_mask_metrics(train_rows + eval_rows, tokenizer, context_length)
    if (
        manifest_fraction is not None
        and abs(metrics["supervised_token_fraction"] - float(manifest_fraction)) > 0.01
    ):
        raise ValueError("assistant supervision fraction disagrees with the data manifest")
    return metrics


def _checkpoint_smoke_subprocess(
    adapter_path: Path,
    *,
    base_model: str,
    revision: Optional[str],
    expect_tied: bool = False,
) -> None:
    script = """
import sys, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
base, adapter, revision, tied = sys.argv[1:]
revision = revision or None
from boldt_posttrain.training import load_tokenizer
tokenizer = load_tokenizer(base, revision=revision)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, revision=revision), adapter)
encoded = tokenizer('Kurzer Reload-Test', return_tensors='pt')
with torch.inference_mode():
    logits = model(**encoded).logits
if not torch.isfinite(logits).all():
    raise SystemExit('non-finite logits after reload')
if tied == '1':
    inner = model.get_base_model()
    if inner.get_input_embeddings().weight.data_ptr() != inner.get_output_embeddings().weight.data_ptr():
        raise SystemExit('weight tying was not preserved')
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            base_model,
            str(adapter_path),
            revision or "",
            "1" if expect_tied else "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode:
        raise RuntimeError(
            "fresh-process checkpoint reload failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _verify_modules_to_save(adapter_path: Path, modules: Sequence[str]) -> None:
    if not modules:
        return
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = set(config.get("modules_to_save") or [])
    if not set(modules) <= configured and not config.get("ensure_weight_tying"):
        raise RuntimeError("saved CPT adapter config is missing expected modules_to_save")
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        raise RuntimeError("CPT modules_to_save verification requires a safetensors adapter")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("checkpoint verification requires safetensors") from exc
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    missing = [module for module in modules if not any(module in key for key in keys)]
    if missing:
        raise RuntimeError("CPT checkpoint is missing saved module weights: " + ", ".join(missing))


def _cpt_peft_and_optimizer(
    model: Any,
    training_cfg: Dict[str, Any],
    cpt_cfg: Mapping[str, Any],
) -> tuple[Any, Any, Any, List[str], bool]:
    requested_embeddings = bool(cpt_cfg.get("train_embeddings", False))
    requested_head = bool(cpt_cfg.get("train_lm_head", False))
    if not (requested_embeddings or requested_head):
        return model, make_peft_config(training_cfg), None, [], False
    if training_cfg.get("method") == "qlora":
        raise ValueError("CPT embedding/lm_head training is not supported with QLoRA")
    try:
        import torch
        from peft import get_peft_model
    except ImportError as exc:
        raise RuntimeError("CPT module training requires torch and peft") from exc
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    named_modules = dict(model.named_modules())
    input_names = [name for name, module in named_modules.items() if module is input_module]
    output_names = [name for name, module in named_modules.items() if module is output_module]
    if requested_embeddings and not input_names:
        raise ValueError("could not resolve the model input-embedding module name")
    if requested_head and not output_names:
        raise ValueError("could not resolve the model output-head module name")
    modules = []
    if requested_embeddings:
        modules.append(input_names[0].split(".")[-1])
    if requested_head:
        modules.append(output_names[0].split(".")[-1])
    modules = list(dict.fromkeys(modules))
    tied = bool(getattr(model.config, "tie_word_embeddings", False)) or (
        input_module.weight.data_ptr() == output_module.weight.data_ptr()
    )
    peft = make_peft_config(
        training_cfg,
        modules_to_save=modules,
        ensure_weight_tying=tied or requested_embeddings,
    )
    model = get_peft_model(model, peft)
    module_parameters = []
    lora_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "modules_to_save" in name:
            module_parameters.append(parameter)
        else:
            lora_parameters.append(parameter)
    if not module_parameters or not lora_parameters:
        raise RuntimeError("CPT optimizer could not separate LoRA and saved-module parameters")
    learning_rate = float(training_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": learning_rate},
            {
                "params": module_parameters,
                "lr": learning_rate * float(cpt_cfg["module_learning_rate_multiplier"]),
            },
        ]
    )
    return model, None, optimizer, modules, tied


def load_trainable_adapter(model: Any, adapter_path: Path) -> Any:
    """Continue a rung from its own adapter with a deliberately fresh optimizer."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("adapter continuation requires peft") from exc
    continued = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    trainable = sum(
        parameter.numel() for parameter in continued.parameters() if parameter.requires_grad
    )
    if trainable <= 0:
        raise RuntimeError("continued adapter has no trainable parameters")
    return continued


def compare_sft_rlvr(
    *, sft_run: Any, rlvr_run: Any, model_start: str, prompt_group: str, gpu_minutes: float
) -> Dict[str, Any]:
    """Run equal-budget callbacks and compare quality and quality per GPU minute."""
    if gpu_minutes <= 0:
        raise ValueError("comparison GPU budget must be positive")
    common = {
        "model_start": model_start,
        "prompt_group": prompt_group,
        "gpu_minutes": gpu_minutes,
        "profile": "proxy",
    }
    sft = dict(sft_run(**common))
    rlvr = dict(rlvr_run(**common))
    for name, result in (("sft", sft), ("rlvr", rlvr)):
        if result.get("status") not in {"ok", "pass"}:
            raise RuntimeError(f"{name} efficiency run failed")
        for metric in ("quality", "gpu_minutes", "peak_vram", "tokens"):
            if not isinstance(result.get(metric), (int, float)):
                raise ValueError(f"{name} result is missing measured {metric}")
        result["quality_per_gpu_minute"] = result["quality"] / max(1, result["gpu_minutes"])
    quality_winner = "sft" if sft["quality"] >= rlvr["quality"] else "rlvr"
    efficiency_winner = (
        "sft" if sft["quality_per_gpu_minute"] >= rlvr["quality_per_gpu_minute"] else "rlvr"
    )
    return {
        "status": "ok",
        "model_start": model_start,
        "prompt_group": prompt_group,
        "budget_gpu_minutes": gpu_minutes,
        "sft": sft,
        "rlvr": rlvr,
        "quality_delta_rlvr_minus_sft": rlvr["quality"] - sft["quality"],
        "reward_development": rlvr.get("reward_development", []),
        "quality_winner": quality_winner,
        "efficiency_winner": efficiency_winner,
    }


def compare_liger(*, run: Any, token_budget: int) -> Dict[str, Any]:
    if token_budget <= 0:
        raise ValueError("Liger comparison requires a positive equal token budget")
    results = {}
    for enabled in (False, True):
        measured = dict(run(use_liger_kernel=enabled, token_budget=token_budget))
        if measured.get("status") not in {"ok", "pass"}:
            raise RuntimeError(f"Liger benchmark failed with use_liger_kernel={enabled}")
        if any(
            not isinstance(measured.get(key), (int, float))
            for key in ("tokens_per_second", "peak_vram", "loss", "tokens")
        ):
            raise ValueError("Liger benchmark returned incomplete measured metrics")
        if measured["tokens"] != token_budget:
            raise ValueError("Liger benchmark did not use the equal token budget")
        results["liger" if enabled else "default"] = measured
    return {"status": "ok", "token_budget": token_budget, **results}


def _train_real(
    *,
    cfg: Dict[str, Any],
    kind: str,
    data_dir: Path,
    out_dir: Path,
    deadline: float,
    device: str,
    parent_adapter: Optional[Path] = None,
    source_group: Optional[str] = None,
    mix_plan_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the supported trainer on streaming manifest shards."""
    try:
        import torch
        from datasets import Dataset, interleave_datasets
        from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("real training requires the train extra") from exc
    from .preference import (
        DeadlineCallback as RecipeDeadlineCallback,
        bind_training_arguments_device,
        run_response_suppression_probes,
        train_preference,
        validate_preference_rows,
    )

    training_cfg = dict(cfg["training"])
    max_steps = training_cfg.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("streaming training requires a positive training.max_steps")
    from .data_pipeline import load_manifest_rows, verify_trainable_manifest

    manifest = verify_trainable_manifest(
        data_dir / "manifest.json", expected_policy_hash=cfg.get("policy_hash")
    )
    if mix_plan_path is not None and kind != "specialist":
        raise ValueError("mix plans currently apply only to SFT specialist training")
    schema = "sft" if kind in {"specialist", "sft"} else kind
    train_rows = load_manifest_rows(data_dir / "manifest.json", schema, split="train")
    eval_rows = load_manifest_rows(data_dir / "manifest.json", schema, split="validation")
    if source_group is not None:
        train_rows = [row for row in train_rows if row.get("source_group") == source_group]
        eval_rows = [row for row in eval_rows if row.get("source_group") == source_group]
    fraction = float(cfg.get("data", {}).get("validation_fraction", {}).get(schema, 0.0))
    if not train_rows:
        raise ValueError(f"manifest has no train rows for {kind}")
    if fraction > 0 and not eval_rows:
        raise ValueError(f"manifest has no validation rows for {kind}")
    tokenizer = load_tokenizer(training_cfg["base_model"], revision=training_cfg.get("revision"))
    if (
        schema == "cpt"
        and training_cfg.get("method") == "qlora"
        and any(cfg.get("cpt", {}).get(flag) for flag in ("train_embeddings", "train_lm_head"))
    ):
        raise ValueError("CPT embedding/lm_head training is not supported with QLoRA")
    model_config = AutoConfig.from_pretrained(
        training_cfg["base_model"], revision=training_cfg.get("revision")
    )
    validate_liger(
        bool(training_cfg.get("use_liger_kernel", False)),
        model_type=str(model_config.model_type),
    )
    assistant_metrics = {
        "enabled": False,
        "supervised_token_fraction": 0.0,
        "examples_without_supervised_tokens": 0,
        "truncated_without_supervision": 0,
    }
    if schema == "sft" and bool(training_cfg.get("assistant_only_loss", False)):
        manifest_fraction = (
            manifest.get("token_statistics", {}).get("sft", {}).get("supervised_token_fraction")
        )
        assistant_metrics = validate_assistant_supervision(
            train_rows,
            eval_rows,
            tokenizer,
            context_length=int(training_cfg["context_length"]),
            manifest_fraction=manifest_fraction,
        )
    preference_diagnostics: Dict[str, Any] = {}
    if schema == "preference":
        preference_cfg = cfg.get("preference", {})
        preference_diagnostics = validate_preference_rows(
            train_rows + eval_rows,
            tokenizer,
            max_prompt_length=int(preference_cfg["max_prompt_length"]),
            max_completion_length=int(preference_cfg["max_completion_length"]),
            context_length=int(training_cfg["context_length"]),
            length_ratio_max=float(preference_cfg["length_ratio_max"]),
        )
    mix_counts: Dict[str, int] = {}
    if mix_plan_path is None:
        dataset = Dataset.from_list(train_rows)
    else:
        from .data_pipeline import verify_hashed_artifact

        mix_document = json.loads(Path(mix_plan_path).read_text(encoding="utf-8"))
        if not verify_hashed_artifact(mix_document):
            raise ValueError("mix plan artifact hash is invalid")
        weights = mix_document.get("weights", {})
        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in train_rows:
            if row.get("source_group") in weights:
                grouped_rows.setdefault(str(row["source_group"]), []).append(row)
        ordered = [
            group for group in sorted(weights) if grouped_rows.get(group) and weights[group] > 0
        ]
        if not ordered:
            raise ValueError("mix plan selects no available source group")
        datasets = [Dataset.from_list(grouped_rows[group]) for group in ordered]
        probabilities = [weights[group] for group in ordered]
        probability_sum = sum(probabilities)
        dataset = interleave_datasets(
            datasets,
            probabilities=[value / probability_sum for value in probabilities],
            seed=int(training_cfg.get("seed", 17)),
            stopping_strategy="first_exhausted",
        )

        def count_mix(row: Dict[str, Any]) -> Dict[str, Any]:
            group = str(row.get("source_group"))
            rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)
            tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            mix_counts[group] = mix_counts.get(group, 0) + tokens
            return row

        dataset = dataset.map(count_mix)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    quantization = None
    if training_cfg.get("method") == "qlora":
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
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg["base_model"],
        revision=training_cfg.get("revision"),
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map=device_map,
    )
    optimizer = None
    cpt_modules: List[str] = []
    cpt_tied = False
    if parent_adapter is not None:
        if schema == "cpt" and any(
            cfg.get("cpt", {}).get(flag) for flag in ("train_embeddings", "train_lm_head")
        ):
            raise ValueError("CPT saved-module training does not support adapter continuation")
        model = load_trainable_adapter(model, parent_adapter)
        peft_config = None
    elif schema == "cpt":
        model, peft_config, optimizer, cpt_modules, cpt_tied = _cpt_peft_and_optimizer(
            model, training_cfg, cfg.get("cpt", {})
        )
    else:
        peft_config = make_peft_config(training_cfg)
    cpt_learning_rates = (
        [float(group["lr"]) for group in optimizer.param_groups] if optimizer is not None else []
    )
    interval = evaluation_interval(max_steps)
    has_validation = eval_dataset is not None
    common = {
        "max_steps": max_steps,
        "learning_rate": float(training_cfg["learning_rate"]),
        "bf16": device.startswith("cuda:"),
        "logging_steps": int(training_cfg.get("logging_steps", 1)),
        "eval_strategy": "steps" if has_validation else "no",
        "save_strategy": "steps" if has_validation else "no",
        "eval_steps": interval if has_validation else None,
        "save_steps": interval if has_validation else 500,
        "save_total_limit": 1 if has_validation else None,
        "load_best_model_at_end": has_validation,
        "metric_for_best_model": "eval_loss" if has_validation else None,
        "greater_is_better": False if has_validation else None,
        "report_to": "none",
        "use_cpu": device == "cpu",
        "per_device_train_batch_size": int(training_cfg.get("batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "use_liger_kernel": bool(training_cfg.get("use_liger_kernel", False)),
        "include_num_input_tokens_seen": True,
        "seed": int(training_cfg["seed"]),
    }
    remaining = deadline - time.monotonic()
    reserve_seconds = min(300.0, max(15.0, remaining * 0.10))
    if remaining <= reserve_seconds:
        raise RuntimeError("training budget is too small for validation, save, and reload reserve")
    deadline_callback = RecipeDeadlineCallback(deadline, reserve_seconds=reserve_seconds)
    trainer_dir = out_dir / "trainer"
    adapter_path = out_dir / "adapter"
    adapter_staging = out_dir / ".adapter-staging"
    started = time.monotonic()
    if kind == "preference":
        method = str(cfg.get("preference", {}).get("method", "dpo"))
        if method == "orpo" and training_cfg.get("use_liger_kernel"):
            raise ValueError("the locked ORPO trainer has no enabled Liger path")
        result = train_preference(
            method=method,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            eval_dataset=eval_dataset,
            output_dir=trainer_dir,
            training_args=common,
            preference_config=cfg.get("preference", {}),
            peft_config=peft_config,
            deadline=deadline,
            reserve_seconds=reserve_seconds,
            device=device,
        )
        trainer = result.trainer
        align_quantized_output_head_input(trainer.model)
        metrics = dict(getattr(result, "metrics", {}))
        probe_path = ROOT / cfg.get("preference", {}).get(
            "probe_set", "data/eval/preference_probes.json"
        )
        probes = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_metrics = run_response_suppression_probes(
            trainer.model,
            tokenizer,
            probes,
            device=device,
            max_new_tokens=int(cfg.get("preference", {}).get("probe_max_new_tokens", 64)),
        )
        metrics.update({f"probe_{key}": value for key, value in probe_metrics.items()})
        if probe_metrics["empty_output_rate"] > float(
            cfg.get("eval", {}).get("empty_output_max", 0.01)
        ):
            raise RuntimeError("response-suppression probe exceeded empty-output gate")
    else:
        args = SFTConfig(
            output_dir=str(trainer_dir),
            packing=bool(training_cfg.get("packing", False)),
            max_length=int(training_cfg["context_length"]),
            assistant_only_loss=(
                bool(training_cfg.get("assistant_only_loss", False)) if schema == "sft" else False
            ),
            **common,
        )
        bind_training_arguments_device(args, device)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[deadline_callback],
            optimizers=(optimizer, None) if optimizer is not None else (None, None),
        )
        result = trainer.train()
        metrics = dict(result.metrics)
    history = list(trainer.state.log_history)
    evaluations = [row for row in history if isinstance(row.get("eval_loss"), (int, float))]
    if has_validation and not evaluations:
        explicit = trainer.evaluate()
        if isinstance(explicit.get("eval_loss"), (int, float)):
            evaluations.append(dict(explicit) | {"step": trainer.state.global_step})
    if has_validation and not evaluations:
        raise RuntimeError("validation is enabled but the trainer produced no validation loss")
    trainer.save_model(str(adapter_staging))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(adapter_staging))
    _verify_modules_to_save(adapter_staging, cpt_modules)
    _checkpoint_smoke_subprocess(
        adapter_staging,
        base_model=training_cfg["base_model"],
        revision=training_cfg.get("revision"),
        expect_tied=cpt_tied,
    )
    adapter_staging.replace(adapter_path)
    shutil.rmtree(trainer_dir, ignore_errors=True)
    stop_reason = (
        deadline_callback.stop_reason
        or getattr(result, "stop_reason", None)
        or ("max_steps" if trainer.state.global_step >= max_steps else "epochs_completed")
    )
    metrics.update(
        {
            "status": "succeeded",
            "steps": int(trainer.state.global_step),
            "stop_reason": stop_reason,
            "eval_loss": evaluations[-1]["eval_loss"] if evaluations else None,
            "best_eval_loss": (
                float(trainer.state.best_metric)
                if isinstance(trainer.state.best_metric, (int, float))
                else min((float(row["eval_loss"]) for row in evaluations), default=None)
            ),
            "best_step": next(
                (
                    int(row.get("step", 0))
                    for row in evaluations
                    if float(row["eval_loss"])
                    == min(float(value["eval_loss"]) for value in evaluations)
                ),
                0,
            ),
            "validation_examples": len(eval_rows),
            "validation_evaluations": len(evaluations),
            "assistant_supervision": assistant_metrics,
            "preference_diagnostics": preference_diagnostics,
            "use_liger_kernel": bool(training_cfg.get("use_liger_kernel", False)),
            "use_rslora": bool(training_cfg.get("use_rslora", False)),
            "cpt_modules_to_save": cpt_modules,
            "cpt_module_learning_rates": cpt_learning_rates,
        }
    )
    metrics["gpu_seconds"] = time.monotonic() - started
    metrics["peak_vram"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda:") else 0
    )
    metrics["trainable_parameters"] = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    metrics["tokens_seen"] = metrics.get("num_input_tokens_seen", 0)
    metrics["tokens_per_second"] = metrics.get(
        "train_tokens_per_second",
        metrics["tokens_seen"] / max(metrics["gpu_seconds"], 1e-9),
    )
    metrics["checkpoint_bytes"] = sum(
        path.stat().st_size for path in adapter_path.rglob("*") if path.is_file()
    )
    if mix_plan_path is not None:
        metrics["mix_tokens_by_group"] = dict(sorted(mix_counts.items()))
        metrics["mix_repeat_counts"] = {group: 0 for group in sorted(mix_counts)}
    return metrics


def run_training_trial(
    *,
    cfg: Dict[str, Any],
    kind: str,
    specialist: str,
    out_root: Path,
    budget_minutes: int,
    argv: List[str],
    dry_run: bool,
    allow_gpu: bool,
    allow_checkpoints: bool,
    data_dir: Path,
    config_errors: List[str],
    device: str = "cuda:0",
    mix_plan: Optional[Path] = None,
) -> int:
    run_id = f"{specialist}-{kind}-{'dry' if dry_run else 'real'}-{prov.stamp()}"
    out_dir = Path(out_root) / run_id
    git = recipe.persist_inputs(out_dir, cfg, argv)
    command = "python " + " ".join(argv)
    plan = _training_plan(cfg, kind, specialist)

    def finish(metrics: Dict[str, Any], status: str, extra: Dict[str, Any]) -> int:
        metrics_doc = {
            "run_id": run_id,
            "status": status,
            "mode": "dry_run" if dry_run else "real",
            "budget_minutes": budget_minutes,
            "git": git,
            "training_plan": plan,
            "metrics": metrics,
            **extra,
        }
        recipe.write_json(out_dir / "metrics.json", metrics_doc)
        output_artifacts = [str(out_dir / "metrics.json")]
        if (out_dir / "adapter").is_dir():
            output_artifacts.append(str(out_dir / "adapter"))
        card = prov.new_run_card(
            run_id,
            _RUN_TYPE[kind],
            command,
            model=plan.get("base_model"),
            metrics=metrics,
            data_manifest=str(data_dir / "manifest.json"),
            input_artifacts=[str(p) for p in (data_dir / "manifest.json",)],
            output_artifacts=output_artifacts,
            notes=extra.get("message", f"{kind} {specialist} ({metrics_doc['mode']})"),
        )
        prov.write_run_card(card, out_dir)
        print(
            json.dumps(
                {
                    "status": status,
                    "mode": metrics_doc["mode"],
                    "run_id": run_id,
                    "out": str(out_dir),
                    **{k: extra[k] for k in ("message",) if k in extra},
                },
                ensure_ascii=False,
            )
        )
        return 0 if status in ("ok", "pass") else 4

    if config_errors:
        return finish(
            recipe.metrics_skeleton(cfg),
            "fail",
            {"message": "config invalid: " + "; ".join(config_errors)},
        )

    if dry_run:
        return finish(
            recipe.metrics_skeleton(cfg),
            "ok",
            {
                "scale_disclaimer": "dry-run plumbing only — no checkpoint, no metrics",
                "message": f"planned {kind} '{specialist}'; pass --real --allow-gpu to train",
            },
        )

    # --- real path -------------------------------------------------------------------------
    if not allow_gpu:
        return finish(
            recipe.metrics_skeleton(cfg),
            "fail",
            {"message": "--real requires --allow-gpu (human hardware gate)"},
        )
    stack_err = recipe.require_real_stack()
    if stack_err:
        return finish(recipe.metrics_skeleton(cfg), "fail", {"message": stack_err})
    if not allow_checkpoints:
        return finish(
            recipe.metrics_skeleton(cfg),
            "fail",
            {"message": "real training requires --allow-checkpoints"},
        )
    if kind == "preference" and not cfg.get("preference", {}).get("enabled", False):
        return finish(
            recipe.metrics_skeleton(cfg),
            "fail",
            {"message": "preference lever is disabled by preference.enabled=false"},
        )
    clean = _manifest_clean(
        data_dir,
        expected_decontamination_hash=cfg.get("data", {}).get("decontamination_hash"),
        expected_policy_hash=cfg.get("policy_hash"),
    )
    if not clean["clean"]:
        return finish(
            recipe.metrics_skeleton(cfg),
            "fail",
            {"message": "data not trainable: " + clean["reason"]},
        )
    try:
        minimum_vram = float(cfg.get("hardware", {}).get("minimum_vram_gb", 40))
        validate_device(device, minimum_vram_gb=minimum_vram)
        metrics = _train_real(
            cfg=cfg,
            kind=kind,
            data_dir=data_dir,
            out_dir=out_dir,
            deadline=recipe.deadline_after(budget_minutes),
            device=device,
            mix_plan_path=resolve_mix_plan(mix_plan),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return finish(
            recipe.metrics_skeleton(cfg),
            "fail",
            {"message": f"technical training failure: {type(exc).__name__}: {exc}"},
        )
    adapter = out_dir / "adapter"
    if not adapter.is_dir() or not any(adapter.iterdir()):
        return finish(metrics, "fail", {"message": "trainer produced no adapter checkpoint"})
    return finish(metrics, "ok", {"message": f"trained {kind} adapter on {device}"})


from .secure_compat.training import (  # noqa: E402, F401
    DeadlineCallback,
    TrainingError,
    collect_model_metadata,
    create_model_and_tokenizer,
    doctor,
    train_adapter,
    validate_target_modules,
    validate_tokenizer,
)
