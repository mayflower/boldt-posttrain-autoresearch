"""Strict loading for the immutable human-owned post-training policy."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "configs" / "posttrain" / "policy.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HUB_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class PolicyError(ValueError):
    """Policy is absent, malformed, movable, or internally inconsistent."""


def _reject_constant(value: str) -> None:
    raise PolicyError(f"non-finite JSON number is forbidden: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(value: Any, location: str = "policy") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyError(f"{location} must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{location}[{index}]")


def _require_keys(document: dict[str, Any], required: set[str], location: str) -> None:
    missing = sorted(required - set(document))
    unknown = sorted(set(document) - required)
    if missing or unknown:
        details = [
            *(f"missing {location}.{key}" for key in missing),
            *(f"unknown {location}.{key}" for key in unknown),
        ]
        raise PolicyError("; ".join(details))


@dataclasses.dataclass(frozen=True)
class Policy:
    path: Path
    document: dict[str, Any]

    @property
    def seed_model(self) -> dict[str, Any]:
        return self.document["seed_model"]

    @property
    def integrity(self) -> dict[str, Any]:
        return self.document["integrity"]

    def to_dict(self) -> dict[str, Any]:
        return self.document


def validate_policy(document: dict[str, Any]) -> None:
    top = {
        "schema_version",
        "artifact_schema_version",
        "seed_model",
        "data",
        "evaluation",
        "scoring",
        "training",
        "merge",
        "integrity",
    }
    _require_keys(document, top, "policy")
    if document["schema_version"] != 1 or document["artifact_schema_version"] != 1:
        raise PolicyError("policy and artifact schema versions must both be 1")
    _finite(document)
    seed = document["seed_model"]
    if not isinstance(seed, dict):
        raise PolicyError("policy.seed_model must be an object")
    _require_keys(
        seed,
        {
            "repo_id",
            "revision",
            "architecture",
            "model_type",
            "dtype",
            "context_length",
            "model_config_sha256",
            "tokenizer_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "special_tokens",
        },
        "policy.seed_model",
    )
    if not HUB_REVISION_RE.fullmatch(seed["revision"]):
        raise PolicyError("policy.seed_model.revision must be an exact 40-character commit SHA")
    for key in (
        "model_config_sha256",
        "tokenizer_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
    ):
        if not SHA256_RE.fullmatch(seed[key]):
            raise PolicyError(f"policy.seed_model.{key} must be a SHA-256 digest")
    if (
        not isinstance(document["evaluation"].get("required_metrics"), list)
        or not document["evaluation"]["required_metrics"]
    ):
        raise PolicyError("policy.evaluation.required_metrics must be non-empty")
    editable = document["integrity"].get("editable_globs")
    protected = document["integrity"].get("protected_globs")
    if not isinstance(editable, list) or not isinstance(protected, list):
        raise PolicyError("policy.integrity globs must be arrays")
    if (
        "configs/posttrain/policy.json" not in protected
        or "configs/posttrain/base.json" not in protected
    ):
        raise PolicyError("policy and base config must protect themselves")
    _require_keys(
        document["data"],
        {
            "allowed_organizations",
            "allowed_licenses",
            "license_reviews_path",
            "remote_code_allowlist",
            "max_rows_per_source",
            "max_total_rows",
            "max_cpt_fraction",
            "max_cpt_tokens",
            "min_german_confidence",
            "language_model",
            "exact_dedup",
            "near_dedup_jaccard",
            "leakage_jaccard",
        },
        "policy.data",
    )
    _require_keys(
        document["data"]["language_model"],
        {"path", "sha256"},
        "policy.data.language_model",
    )
    evaluation = document["evaluation"]
    _require_keys(
        evaluation,
        {
            "suite_id",
            "suite_path",
            "required_metrics",
            "lm_eval_tasks",
            "task_revisions",
            "decoding",
            "bootstrap",
        },
        "policy.evaluation",
    )
    _require_keys(
        evaluation["decoding"],
        {"do_sample", "temperature", "max_new_tokens", "seed"},
        "policy.evaluation.decoding",
    )
    _require_keys(
        evaluation["bootstrap"],
        {"samples", "confidence", "seed"},
        "policy.evaluation.bootstrap",
    )
    if set(evaluation["task_revisions"]) != set(evaluation["lm_eval_tasks"]):
        raise PolicyError("policy evaluation task revisions must exactly match task names")
    for task, descriptor in evaluation["task_revisions"].items():
        _require_keys(
            descriptor,
            {"dataset_id", "config", "split", "revision"},
            f"policy.evaluation.task_revisions.{task}",
        )
        if not HUB_REVISION_RE.fullmatch(descriptor["revision"]):
            raise PolicyError(f"policy task {task} must use an exact dataset commit")
    _require_keys(document["scoring"], {"weights", "promotion"}, "policy.scoring")
    _require_keys(
        document["scoring"]["weights"],
        {
            "german_instruction",
            "format_following",
            "reasoning_core",
            "longcontext",
            "lm_eval_regression",
            "english_bleed",
            "response_suppression",
            "safety_regression",
        },
        "policy.scoring.weights",
    )
    _require_keys(
        document["scoring"]["promotion"],
        {
            "min_score",
            "german_instruction_min_delta",
            "format_following_min_delta",
            "reasoning_core_min_delta",
            "longcontext_min_delta",
            "safety_min_delta",
            "lm_eval_regression_tolerance",
            "english_bleed_max",
            "empty_output_max",
            "refusal_spike_max",
            "over_refusal_spike_max",
            "max_leakage_hits",
        },
        "policy.scoring.promotion",
    )
    _require_keys(
        document["training"],
        {
            "allowed_methods",
            "allowed_specialists",
            "qlora",
            "cpt_max_learning_rate",
            "assistant_only_loss_required",
        },
        "policy.training",
    )
    _require_keys(
        document["training"]["qlora"],
        {"quant_type", "double_quant", "compute_dtype"},
        "policy.training.qlora",
    )
    _require_keys(
        document["merge"],
        {"allowed_methods", "max_candidates", "parameter_bounds"},
        "policy.merge",
    )
    _require_keys(
        document["merge"]["parameter_bounds"],
        {"weight_min", "weight_max", "density_min", "density_max"},
        "policy.merge.parameter_bounds",
    )
    _require_keys(
        document["integrity"],
        {"editable_globs", "protected_globs"},
        "policy.integrity",
    )
    _require_keys(
        seed["special_tokens"],
        {"bos_token", "eos_token", "pad_token"},
        "policy.seed_model.special_tokens",
    )
    for location in (
        document["data"]["allowed_organizations"],
        document["data"]["allowed_licenses"],
        evaluation["required_metrics"],
        evaluation["lm_eval_tasks"],
        document["training"]["allowed_methods"],
        document["training"]["allowed_specialists"],
        document["merge"]["allowed_methods"],
        editable,
        protected,
    ):
        if not isinstance(location, list) or not all(isinstance(item, str) for item in location):
            raise PolicyError("policy string-list field has an invalid value")


def load_policy(path: str | Path = DEFAULT_POLICY) -> Policy:
    resolved = Path(path).resolve()
    try:
        document = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"cannot read policy {resolved}: {exc}") from exc
    if not isinstance(document, dict):
        raise PolicyError("policy must be a JSON object")
    validate_policy(document)
    return Policy(path=resolved, document=document)
