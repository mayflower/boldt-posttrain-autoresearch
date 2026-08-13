"""Config resolution for the post-training loop (pure stdlib).

A config may declare ``"extends": "<path-relative-to-repo-root>"``; ``resolve_config`` deep-merges
the base under the overlay so ``configs/posttrain/current.json`` inherits ``base.json`` defaults.
The merged dict records ``_extends`` for provenance. ``validate_config_dict`` performs cheap,
fail-closed structural checks used by ``pt_status`` and the dry-run paths — it never imports ML.
"""

from __future__ import annotations

import json
import hashlib
import pathlib
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "posttrain" / "current.json"
BASE_CONFIG = ROOT / "configs" / "posttrain" / "base.json"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; nested dicts merged)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(
    config_path: pathlib.Path, _seen: Optional[set[pathlib.Path]] = None
) -> Dict[str, Any]:
    """Load a config, merging the base referenced via ``extends`` (path relative to repo root)."""
    config_path = pathlib.Path(config_path)
    config_path = config_path if config_path.is_absolute() else (ROOT / config_path)
    seen = set(_seen or set())
    resolved_path = config_path.resolve()
    if resolved_path in seen:
        raise ValueError(f"cyclic config inheritance at {config_path}")
    seen.add(resolved_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    extends = cfg.get("extends")
    if isinstance(extends, str):
        base = resolve_config(ROOT / extends, seen)
        merged = deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
        merged["_extends"] = extends
        cfg = merged
    policy_path = cfg.get("policy", "configs/posttrain/policy.json")
    path = ROOT / policy_path
    if path.is_file():
        policy = json.loads(path.read_text(encoding="utf-8"))
        cfg["policy"] = policy_path
        cfg["policy_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
        cfg["promotion_gates"] = policy.get("promotion_gates", {})
        if isinstance(cfg.get("data"), dict):
            cfg["data"]["allowed_licenses"] = policy.get("allowed_licenses", [])
            cfg["data"]["allowed_sources"] = policy.get("allowed_sources", [])
    return cfg


def validate_config_dict(cfg: Dict[str, Any]) -> List[str]:
    """Cheap structural validation of a RESOLVED config. Returns a list of human-readable errors
    (empty == valid). Fail-closed: missing blocks are errors, not silently defaulted."""
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["config must be a JSON object"]

    training = cfg.get("training")
    if not isinstance(training, dict):
        errors.append("missing 'training' block")
    else:
        if not training.get("base_model"):
            errors.append("training.base_model is required (the protected Boldt seed)")
        method = training.get("method")
        if method not in ("qlora", "lora", "full", None):
            errors.append(f"training.method '{method}' not in qlora|lora|full")
        if not isinstance(training.get("seed"), int):
            errors.append("training.seed must be an integer")
        for flag in ("assistant_only_loss", "use_liger_kernel", "use_rslora"):
            if not isinstance(training.get(flag), bool):
                errors.append(f"training.{flag} must be boolean")

    data = cfg.get("data")
    if not isinstance(data, dict):
        errors.append("missing 'data' block")
    else:
        sources = data.get("sources", [])
        if data.get("org") != "openeurollm" and not sources:
            errors.append("non-default data.org requires an explicit pinned data.sources allowlist")
        if sources:
            if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
                errors.append("data.sources must be a list of source objects")
            else:
                allowed = {
                    (
                        str(item.get("dataset", "")),
                        str(item.get("revision", "")),
                        str(item.get("license", "")).lower(),
                    )
                    for item in data.get("allowed_sources", [])
                    if isinstance(item, dict)
                }
                for source in sources:
                    identity = (
                        str(source.get("dataset", "")),
                        str(source.get("revision", "")),
                        str(source.get("license", "")).lower(),
                    )
                    if not all(identity):
                        errors.append("each data source requires dataset, revision, and license")
                    elif identity not in allowed:
                        errors.append(
                            f"data source is not policy-allowed: {identity[0]}@{identity[1]}"
                        )
        if not data.get("language_allowlist"):
            errors.append("data.language_allowlist is required for German filtering")
        maximum = data.get("max_rows")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            errors.append("data.max_rows must be a positive integer")
        fractions = data.get("validation_fraction")
        if not isinstance(fractions, dict):
            errors.append("data.validation_fraction is required")
        else:
            for schema in ("sft", "cpt", "preference", "verified_math"):
                value = fractions.get(schema)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"data.validation_fraction.{schema} must be numeric")
                elif not 0.0 <= float(value) < 0.5:
                    errors.append(
                        f"data.validation_fraction.{schema} must satisfy 0.0 <= value < 0.5"
                    )

    preference = cfg.get("preference")
    if not isinstance(preference, dict):
        errors.append("missing 'preference' block")
    else:
        if not isinstance(preference.get("enabled"), bool):
            errors.append("preference.enabled must be boolean")
        if preference.get("method") not in {"dpo", "kto", "orpo"}:
            errors.append("preference.method must be dpo, kto, or orpo")
        weight = preference.get("sft_loss_weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or float(weight) < 0:
            errors.append("preference.sft_loss_weight must be a non-negative number")
        if preference.get("method") != "dpo" and isinstance(weight, (int, float)) and weight > 0:
            errors.append("preference.sft_loss_weight is supported only for DPO")

    cpt = cfg.get("cpt")
    if not isinstance(cpt, dict):
        errors.append("missing 'cpt' block")
    else:
        multiplier = cpt.get("module_learning_rate_multiplier")
        if (
            not isinstance(multiplier, (int, float))
            or isinstance(multiplier, bool)
            or not 0.0 < float(multiplier) <= 1.0
        ):
            errors.append("cpt.module_learning_rate_multiplier must satisfy 0.0 < value <= 1.0")
        for flag in ("train_embeddings", "train_lm_head"):
            if not isinstance(cpt.get(flag), bool):
                errors.append(f"cpt.{flag} must be boolean")

    verified = cfg.get("verified_rl")
    if not isinstance(verified, dict):
        errors.append("missing 'verified_rl' block")
    else:
        if not isinstance(verified.get("enabled"), bool):
            errors.append("verified_rl.enabled must be boolean")
        if verified.get("reward_profile") != "math_accuracy":
            errors.append("verified_rl.reward_profile must be 'math_accuracy'")
        generations = verified.get("num_generations")
        if not isinstance(generations, int) or isinstance(generations, bool) or generations < 2:
            errors.append("verified_rl.num_generations must be an integer >= 2")
        batch_size = verified.get("batch_size", generations)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            errors.append("verified_rl.batch_size must be a positive integer")
        elif isinstance(generations, int) and generations > 0 and batch_size % generations:
            errors.append("verified_rl.batch_size must be divisible by num_generations")
        for key in ("max_prompt_length", "max_completion_length"):
            value = verified.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"verified_rl.{key} must be a positive integer")
        temperature = verified.get("temperature")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            errors.append("verified_rl.temperature must be positive")
        beta = verified.get("beta")
        if not isinstance(beta, (int, float)) or beta < 0:
            errors.append("verified_rl.beta must be non-negative")
        if isinstance(training, dict):
            prompt_length = verified.get("max_prompt_length")
            completion_length = verified.get("max_completion_length")
            context_length = training.get("context_length")
            if all(
                isinstance(value, int)
                for value in (prompt_length, completion_length, context_length)
            ) and (prompt_length + completion_length > context_length):
                errors.append(
                    "verified_rl prompt and completion lengths exceed training.context_length"
                )

    ev = cfg.get("eval")
    if not isinstance(ev, dict):
        errors.append("missing 'eval' block")
    elif not ev.get("lm_eval_tasks"):
        errors.append("eval.lm_eval_tasks is required for German-core regression checks")

    return errors


def load_resolved(config_path: Optional[str] = None) -> Dict[str, Any]:
    return resolve_config(pathlib.Path(config_path) if config_path else DEFAULT_CONFIG)


from .secure_compat.config import (  # noqa: E402, F401
    ConfigError,
    ExperimentConfig,
    load_experiment,
)
