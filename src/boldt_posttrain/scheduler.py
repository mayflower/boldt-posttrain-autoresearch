"""Serial three-rung Successive Halving for SFT, preference, and RLOO trials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ALLOWED_OVERRIDES = {
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_init",
    "use_rslora",
    "packing",
    "use_liger_kernel",
    "kl_coefficient",
}
ALLOWED_LORA_RANKS = {4, 8, 16, 32, 64}


def validate_search_plan(plan: Mapping[str, Any], *, lever: str = "sft") -> List[Dict[str, Any]]:
    trials = plan.get("trials")
    if not isinstance(trials, list) or not 1 <= len(trials) <= 9:
        raise ValueError("search plan must contain between one and nine trials")
    seen = set()
    result = []
    for trial in trials:
        if not isinstance(trial, dict) or not isinstance(trial.get("trial_id"), str):
            raise ValueError("every search trial requires a string trial_id")
        trial_id = trial["trial_id"]
        if not trial_id or trial_id in seen:
            raise ValueError("trial IDs must be non-empty and unique")
        seen.add(trial_id)
        overrides = trial.get("overrides")
        if not isinstance(overrides, dict) or set(overrides) - ALLOWED_OVERRIDES:
            raise ValueError(f"trial {trial_id} contains unsupported override dimensions")
        if any(isinstance(value, (dict, list, tuple, set)) for value in overrides.values()):
            raise ValueError("search overrides must be scalar values")
        if "lora_r" in overrides and overrides["lora_r"] not in ALLOWED_LORA_RANKS:
            raise ValueError("unsupported LoRA rank")
        if overrides.get("lora_init", "default") not in {"default", "pissa_niter_4"}:
            raise ValueError("unsupported LoRA initializer")
        if lever == "rlvr":
            if set(overrides) - {"learning_rate", "lora_r", "kl_coefficient"}:
                raise ValueError("RLOO search only supports learning_rate, lora_r, kl_coefficient")
            if overrides.get("lora_r", 4) not in {4, 8, 16}:
                raise ValueError("RLOO LoRA rank must be 4, 8, or 16")
        result.append({"trial_id": trial_id, "overrides": dict(overrides)})
    if lever == "rlvr":
        for dimension in ("learning_rate", "kl_coefficient"):
            if len({trial["overrides"].get(dimension) for trial in result}) > 3:
                raise ValueError(f"RLOO {dimension} supports at most three values")
    return result


def continuation_metadata(parent_run_id: Optional[str]) -> Dict[str, Any]:
    return {
        "parent_run_id": parent_run_id,
        "continuation_mode": "adapter_fresh_optimizer" if parent_run_id else "new_adapter",
    }


def _rank(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    valid = [
        dict(result)
        for result in results
        if result.get("status") in {"ok", "pass"}
        and int(result.get("technical_error_count", 0)) == 0
        and result.get("hard_gates_passed") is True
        and isinstance(result.get("proxy_score"), (int, float))
    ]
    return sorted(
        valid,
        key=lambda result: (
            -float(result["proxy_score"]),
            float(result.get("gpu_seconds", float("inf"))),
            str(result["trial_id"]),
        ),
    )


@dataclass(frozen=True)
class Rung:
    index: int
    additional_steps: int
    cumulative_steps: int


def rungs(full_budget: int) -> List[Rung]:
    if full_budget <= 256:
        raise ValueError("full search budget must exceed 256 steps")
    return [Rung(1, 64, 64), Rung(2, 192, 256), Rung(3, full_budget - 256, full_budget)]


def run_successive_halving(
    *,
    plan: Mapping[str, Any],
    full_budget: int,
    train_and_proxy: Callable[
        [Mapping[str, Any], Rung, Optional[Mapping[str, Any]]], Mapping[str, Any]
    ],
    dev_evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    lever: str = "sft",
) -> Dict[str, Any]:
    """Run every trial serially, retaining the best third after each cheap rung."""
    active = validate_search_plan(plan, lever=lever)
    all_results: List[Dict[str, Any]] = []
    previous: Dict[str, Dict[str, Any]] = {}
    rung_docs = []
    for rung in rungs(full_budget):
        results = []
        for trial in active:  # deliberately serial: no worker pool and no background futures
            parent = previous.get(trial["trial_id"])
            try:
                result = dict(train_and_proxy(trial, rung, parent))
            except Exception as exc:
                if type(exc).__name__.lower().startswith("integrity"):
                    raise
                result = {
                    "status": "failed",
                    "technical_error_count": 1,
                    "technical_error": f"{type(exc).__name__}: {exc}",
                }
            result.update(
                {
                    "trial_id": trial["trial_id"],
                    "rung": rung.index,
                    "additional_steps": rung.additional_steps,
                    "cumulative_steps": rung.cumulative_steps,
                    **continuation_metadata(parent.get("run_id") if parent else None),
                }
            )
            results.append(result)
            all_results.append(result)
        ranked = _rank(results)
        if not ranked:
            technically_successful = any(
                result.get("status") in {"ok", "pass"}
                and int(result.get("technical_error_count", 0)) == 0
                for result in results
            )
            return {
                "status": "rejected" if technically_successful else "failed",
                "reason": (
                    "no technically successful trial passed hard gates"
                    if technically_successful
                    else "no technically successful trial"
                ),
                "rungs": rung_docs + [{"rung": rung.index, "results": results}],
                "runs": all_results,
                "winner": None,
            }
        rung_doc = {
            "rung": rung.index,
            "additional_steps": rung.additional_steps,
            "cumulative_steps": rung.cumulative_steps,
            "results": results,
        }
        if rung.index < 3:
            keep = max(1, math.ceil(len(ranked) / 3))
            survivors = ranked[:keep]
            rung_doc["survivors"] = [result["trial_id"] for result in survivors]
            active_by_id = {trial["trial_id"]: trial for trial in active}
            active = [active_by_id[result["trial_id"]] for result in survivors]
            previous = {result["trial_id"]: result for result in survivors}
        rung_docs.append(rung_doc)
    winner = _rank(rung_docs[-1]["results"])[0]
    dev = dict(dev_evaluate(winner))
    if int(dev.get("technical_error_count", 0)):
        return {
            "status": "failed",
            "rungs": rung_docs,
            "runs": all_results,
            "winner": winner,
            "dev_evaluation": dev,
        }
    return {
        "status": "ok",
        "rungs": rung_docs,
        "runs": all_results,
        "winner": winner,
        "dev_evaluation": dev,
    }


def build_mix_plan(
    probes: Sequence[Mapping[str, Any]], *, minimum_weights: Mapping[str, float]
) -> Dict[str, Any]:
    """Convert measured proxy delta/GPU-minute utilities into one normalized mix."""
    if not probes:
        raise ValueError("at least one source probe is required")
    utilities = {}
    for probe in probes:
        group = str(probe["source_group"])
        delta = float(probe["proxy_score_delta"])
        minutes = max(1.0, float(probe["gpu_minutes"]))
        utilities[group] = max(0.0, delta / minutes)
    total = sum(utilities.values())
    weights = {group: (utility / total if total else 0.0) for group, utility in utilities.items()}
    required = {
        group: float(weight) for group, weight in minimum_weights.items() if group in weights
    }
    if sum(required.values()) > 1.0:
        raise ValueError("minimum mix weights exceed one")
    for group, floor in required.items():
        weights[group] = max(weights[group], floor)
    excess = sum(weights.values()) - 1.0
    if excess > 0:
        adjustable = [group for group in weights if group not in required and weights[group] > 0]
        adjustable_total = sum(weights[group] for group in adjustable)
        if adjustable_total < excess - 1e-12:
            raise ValueError("minimum weights leave no feasible normalized mix")
        for group in adjustable:
            weights[group] -= excess * weights[group] / adjustable_total
    elif excess < 0:
        positive = [group for group, weight in weights.items() if weight > 0]
        if not positive:
            raise ValueError("all source probes have non-positive utility")
        positive_total = sum(weights[group] for group in positive)
        for group in positive:
            weights[group] += (-excess) * weights[group] / positive_total
    return {
        "schema_version": 1,
        "status": "ok",
        "utilities": utilities,
        "weights": {group: weights[group] for group in sorted(weights)},
        "probes": [dict(probe) for probe in probes],
    }


def run_mix_probes(
    groups: Sequence[str],
    probe: Callable[[str, int], Mapping[str, Any]],
    *,
    minimum_weights: Mapping[str, float],
) -> Dict[str, Any]:
    """Train and evaluate one serial 64-step adapter for each source group."""
    results = []
    for group in sorted(groups):
        result = dict(probe(group, 64))
        if result.get("status") not in {"ok", "pass"}:
            raise RuntimeError(f"source probe failed for {group}")
        result["source_group"] = group
        results.append(result)
    return build_mix_plan(results, minimum_weights=minimum_weights)
