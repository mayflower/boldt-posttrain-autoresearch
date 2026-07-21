"""Strict experiment configuration loading.

Experiment files contain only mutable research parameters. Human-owned model, data,
evaluation, scoring, promotion, and integrity rules live in ``policy.json``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "posttrain" / "current.json"

FORBIDDEN_KEY_FRAGMENTS = (
    "threshold",
    "tolerance",
    "required_metric",
    "allowed_license",
    "allowed_org",
    "protected_glob",
    "base_model_revision",
    "eval_task",
    "promotion",
    "integrity",
)

_SOURCE_SCHEMA = {
    "dataset_id": str,
    "revision": str,
    "config": str,
    "split": str,
    "schema": str,
}
_SCHEMA: dict[str, Any] = {
    "schema_version": int,
    "experiment": {"name": str, "hypothesis": str, "lever": str},
    "data": {"sources": [_SOURCE_SCHEMA], "max_rows": int},
    "training": {
        "specialist": str,
        "method": str,
        "learning_rate": (int, float),
        "num_train_epochs": (int, float),
        "max_steps": int,
        "warmup_ratio": (int, float),
        "per_device_batch_size": int,
        "gradient_accumulation_steps": int,
        "context_length": int,
        "lora_r": int,
        "lora_alpha": int,
        "lora_dropout": (int, float),
        "target_modules": [str],
        "seed": int,
        "packing": bool,
        "gradient_checkpointing": bool,
        "assistant_only_loss": bool,
        "quantization": str,
    },
    "preference": {
        "enabled": bool,
        "method": str,
        "loss_type": str,
        "beta": (int, float),
        "rpo_alpha": (int, float),
        "length_ratio_max": (int, float),
        "max_prompt_length": int,
        "max_completion_length": int,
    },
    "distillation": {
        "teacher": str,
        "teacher_license": str,
        "max_prompts": int,
        "min_new_tokens": int,
        "max_new_tokens": int,
    },
    "merge": {
        "enabled": bool,
        "inputs": [str],
        "methods": [str],
        "parameters": dict,
        "dtype": str,
    },
    "evaluation": {"batch_size": int},
    "resources": {"budget_minutes": (int, float)},
}


class ConfigError(ValueError):
    """The experiment file violates its immutable schema boundary."""


def _reject_constant(value: str) -> None:
    raise ConfigError(f"non-finite JSON number is forbidden: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read experiment config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("experiment config must be a JSON object")
    return value


def _validate(value: Any, schema: Any, location: str) -> list[str]:
    errors: list[str] = []
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            return [f"{location} must be an object"]
        unknown = sorted(set(value) - set(schema))
        missing = sorted(set(schema) - set(value))
        errors.extend(f"unknown key {location}.{key}" for key in unknown)
        errors.extend(f"missing key {location}.{key}" for key in missing)
        for key in sorted(set(value) & set(schema)):
            errors.extend(_validate(value[key], schema[key], f"{location}.{key}"))
        return errors
    if isinstance(schema, list):
        if not isinstance(value, list):
            return [f"{location} must be an array"]
        return [
            error
            for index, item in enumerate(value)
            for error in _validate(item, schema[0], f"{location}[{index}]")
        ]
    expected = schema if isinstance(schema, tuple) else (schema,)
    if isinstance(value, bool) and bool not in expected:
        return [f"{location} has invalid type bool"]
    if not isinstance(value, expected):
        names = "|".join(item.__name__ for item in expected)
        return [f"{location} must be {names}"]
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{location} must be finite"]
    return []


def _find_forbidden(value: Any, location: str = "config") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                errors.append(f"protected policy key is forbidden in experiments: {location}.{key}")
            errors.extend(_find_forbidden(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_forbidden(child, f"{location}[{index}]"))
    return errors


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    document: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.document


def validate_config_dict(document: dict[str, Any]) -> list[str]:
    errors = _find_forbidden(document)
    errors.extend(_validate(document, _SCHEMA, "config"))
    if document.get("schema_version") != 1:
        errors.append("config.schema_version must be 1")
    return sorted(set(errors))


def load_experiment(path: str | Path = DEFAULT_CONFIG) -> ExperimentConfig:
    resolved = Path(path).resolve()
    document = _load_json(resolved)
    errors = validate_config_dict(document)
    if errors:
        raise ConfigError("; ".join(errors))
    return ExperimentConfig(path=resolved, document=document)


def load_resolved(config_path: str | None = None) -> dict[str, Any]:
    return load_experiment(config_path or DEFAULT_CONFIG).to_dict()


def resolve_config(config_path: Path) -> dict[str, Any]:
    return load_experiment(config_path).to_dict()
