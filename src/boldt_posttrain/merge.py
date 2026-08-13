"""Small serial merge search and proxy-to-dev candidate selection."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Mapping, Sequence


def build_candidates(
    parents: Sequence[Mapping[str, Any]], methods: Sequence[str], *, limit: int
) -> List[Dict[str, Any]]:
    if limit <= 0:
        raise ValueError("merge candidate limit must be positive")
    candidates = []
    for left, right in itertools.combinations(sorted(parents, key=lambda p: str(p["run_id"])), 2):
        if left.get("base_model") != right.get("base_model"):
            continue
        for method in methods:
            run_id = f"{left['run_id']}+{right['run_id']}::{method}"
            candidates.append(
                {
                    "run_id": run_id,
                    "parents": [left["run_id"], right["run_id"]],
                    "method": method,
                    "status": "pending",
                }
            )
            if len(candidates) >= limit:
                return candidates
    return candidates


def rank_proxy_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    eligible = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("technical_error_count", 0) == 0
        and candidate.get("hard_gates_passed") is True
        and isinstance(candidate.get("proxy_score"), (int, float))
    ]
    return sorted(
        eligible,
        key=lambda item: (
            -float(item["proxy_score"]),
            float(item.get("gpu_seconds", float("inf"))),
            str(item["run_id"]),
        ),
    )


def run_merge_round(
    *,
    candidates: Sequence[Mapping[str, Any]],
    proxy_evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    dev_evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Proxy-evaluate every candidate and fully evaluate only the best gate-capable one."""
    evaluated = []
    for candidate in candidates:
        result = dict(proxy_evaluate(candidate))
        evaluated.append({**dict(candidate), **result, "profile": "proxy"})
    ranked = rank_proxy_candidates(evaluated)
    if not ranked:
        return {
            "status": "rejected",
            "winner": None,
            "candidates": evaluated,
            "reason": "no candidate passed proxy hard gates",
        }
    winner = ranked[0]
    dev = dict(dev_evaluate(winner))
    winner = {**winner, "dev_evaluation": dev}
    status = (
        "failed"
        if dev.get("technical_error_count", 0)
        else ("ok" if dev.get("status") in {"ok", "pass"} else "rejected")
    )
    return {"status": status, "winner": winner, "candidates": evaluated}


def mergekit_config(
    *,
    method: str,
    base_model: str,
    models: Sequence[str],
    dtype: str,
    tokenizer_source: str = "union",
) -> Dict[str, Any]:
    """Build a mergekit document for PEFT adapters sharing one exact base model."""
    if method not in {"linear", "slerp", "ties", "dare_ties"} or len(models) != 2:
        raise ValueError("supported merge methods require exactly two models")
    references = [{"model": base_model, "lora": adapter} for adapter in models]
    document: Dict[str, Any] = {
        "merge_method": method,
        "dtype": dtype,
        "tokenizer": {"source": tokenizer_source},
    }
    document["base_model"] = references[0] if method == "slerp" else base_model
    if method == "linear":
        document["models"] = [
            {"model": reference, "parameters": {"weight": 0.5}} for reference in references
        ]
        document["parameters"] = {"normalize": True, "weight": 0.5}
    elif method == "slerp":
        document["models"] = [{"model": reference} for reference in references]
        document["parameters"] = {"t": 0.5}
    else:
        document["models"] = [
            {"model": reference, "parameters": {"weight": 0.5, "density": 0.5}}
            for reference in references
        ]
        document["parameters"] = {
            "normalize": True,
            "int8_mask": True,
            "weight": 0.5,
            "density": 0.5,
        }
    return document


from .secure_compat.merge import (  # noqa: E402, F401
    MergeError,
    MergeInput,
    eligible_input,
    execute_merge,
    materialize_adapter,
    merge_configuration,
    merge_parameter_grid,
    run_search,
    validate_input_set,
    validate_merge_parameters,
)
