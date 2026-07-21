"""Fail-closed scoring over a complete model-to-evaluation artifact chain."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import provenance
from .artifacts import (
    ArtifactRef,
    EventLog,
    atomic_write_json,
    canonical_json_bytes,
    exclusive_lock,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
    verify_artifact_ref,
)
from .evaluation import load_suite, suite_hash
from .policy import Policy, load_policy
from .resolver import OUTPUTS, resolve_candidate

ROOT = Path(__file__).resolve().parents[2]


class ScoringError(RuntimeError):
    """Evaluation or score evidence is incomplete, stale, or manipulated."""


@dataclass(frozen=True)
class VerifiedEvaluation:
    run_dir: Path
    summary_path: Path
    run_card_path: Path
    summary: dict[str, Any]
    run_card: dict[str, Any]
    raw_records: list[dict[str, Any]]
    summary_ref: ArtifactRef


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ScoringError(f"non-finite JSON token {token} in {path}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScoringError(f"expected a JSON object at {path}")
    _finite(value)
    return value


def _finite(value: Any, location: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ScoringError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{location}[{index}]")


def _rate(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ScoringError(f"{location} must be finite and in [0, 1]")
    return result


def _event_successes(outputs_root: Path) -> dict[str, dict[str, Any]]:
    EventLog(outputs_root).validate()
    successes: dict[str, dict[str, Any]] = {}
    for line in (outputs_root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] == "run_finished" and event["payload"].get("status") == "succeeded":
            successes[event["run_id"]] = event["payload"]
    return successes


def _verify_observed_head(card: Mapping[str, Any], outputs_root: Path) -> None:
    observed = card.get("environment", {}).get("event_head")
    if not isinstance(observed, dict) or set(observed) != {
        "sequence",
        "last_event_hash",
        "log_sha256",
    }:
        raise ScoringError("run card has no complete observed event head")
    lines = (outputs_root / "events.jsonl").read_bytes().splitlines(keepends=True)
    sequence = observed["sequence"]
    if not isinstance(sequence, int) or sequence < 1 or sequence > len(lines):
        raise ScoringError("observed event sequence is outside the current chain")
    prefix = b"".join(lines[:sequence])
    event = json.loads(lines[sequence - 1])
    if (
        sha256_bytes(prefix) != observed["log_sha256"]
        or event.get("event_hash") != observed["last_event_hash"]
    ):
        raise ScoringError("observed event head does not match the current chain prefix")


def _card_output(card: Mapping[str, Any], role: str) -> ArtifactRef:
    matches = [ArtifactRef.from_dict(item) for item in card["outputs"] if item.get("role") == role]
    if len(matches) != 1:
        raise ScoringError(f"run card must contain exactly one {role} output")
    return matches[0]


def _read_raw(ref: Mapping[str, Any], repository_root: Path) -> list[dict[str, Any]]:
    path = verify_artifact_ref(ref, root=repository_root)
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ScoringError(f"non-finite raw generation token {token}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ScoringError(f"invalid raw generation line {index}") from exc
        if not isinstance(record, dict):
            raise ScoringError(f"raw generation line {index} is not an object")
        _finite(record, f"raw[{index}]")
        records.append(record)
    return records


def _validate_metrics(summary: Mapping[str, Any], policy: Policy) -> None:
    metrics = summary.get("metrics")
    required = set(policy.document["evaluation"]["required_metrics"])
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ScoringError("evaluation metrics do not exactly match protected requirements")
    for name, value in metrics.items():
        if name == "lm_eval":
            expected_tasks = set(policy.document["evaluation"]["lm_eval_tasks"])
            if not isinstance(value, dict) or set(value) != expected_tasks:
                raise ScoringError("lm-eval task results are incomplete or unexpected")
            for task, score in value.items():
                _rate(score, f"metrics.lm_eval.{task}")
        else:
            _rate(value, f"metrics.{name}")


def verify_evaluation(
    run_dir: Path,
    policy: Policy,
    *,
    outputs_root: Path,
    baseline: bool,
    repository_root: Path,
) -> VerifiedEvaluation:
    summary_path = run_dir / "summary.json"
    card_path = run_dir / "run_card.json"
    summary, card = _load_json(summary_path), _load_json(card_path)
    validate_run_card(card)
    expected_type = "baseline" if baseline else "eval"
    if (
        summary.get("schema_version") != 1
        or summary.get("mode") != "real"
        or summary.get("status") != "succeeded"
        or card["run_type"] != expected_type
        or card["mode"] != "real"
        or card["status"] != "succeeded"
        or summary.get("run_id") != card["run_id"]
        or card["run_id"] != run_dir.name
    ):
        raise ScoringError("evaluation mode, status, type, or identity is invalid")
    if "scale_disclaimer" in summary:
        raise ScoringError("scale-disclaimer evaluations are not scoring evidence")
    policy_hash = sha256_file(policy.path)
    if summary.get("policy_sha256") != policy_hash or card["policy"].get("sha256") != policy_hash:
        raise ScoringError("evaluation policy hash is stale or different")
    if summary.get("suite_hash") != suite_hash() or card["data"].get("suite_hash") != suite_hash():
        raise ScoringError("evaluation suite hash is stale or different")
    task_revisions = policy.document["evaluation"]["task_revisions"]
    if (
        summary.get("task_revisions") != task_revisions
        or card["data"].get("task_revisions") != task_revisions
    ):
        raise ScoringError("evaluation task revisions differ from policy")
    _validate_metrics(summary, policy)
    repository_cases = {case["case_id"]: case for case in load_suite()}
    per_case = summary.get("per_case")
    if not isinstance(per_case, dict) or set(per_case) != set(repository_cases):
        raise ScoringError("per-case evaluation results are incomplete or misaligned")
    for case_id, value in per_case.items():
        _rate(value, f"per_case.{case_id}")
    summary_ref = _card_output(card, "eval_summary")
    if verify_artifact_ref(summary_ref, root=repository_root) != summary_path.resolve():
        raise ScoringError("eval-summary artifact does not point to its run summary")
    raw_ref = _card_output(card, "raw_generations")
    model_ref = _card_output(card, "resolved_model")
    lm_eval_ref = _card_output(card, "lm_eval_results")
    for ref in (raw_ref, model_ref, lm_eval_ref):
        verify_artifact_ref(ref, root=repository_root)
    if summary.get("raw_generations") != raw_ref.to_dict():
        raise ScoringError("summary raw-generations reference differs from run card")
    if summary.get("model_artifact") != model_ref.to_dict():
        raise ScoringError("summary model reference differs from run card")
    if summary.get("lm_eval_artifact") != lm_eval_ref.to_dict():
        raise ScoringError("summary lm-eval reference differs from run card")
    stored_model = _load_json(verify_artifact_ref(model_ref, root=repository_root))
    if stored_model != summary.get("resolved_model") or stored_model != card.get("model"):
        raise ScoringError("resolved model differs across evaluation artifacts")
    records = _read_raw(raw_ref.to_dict(), repository_root)
    by_id = {record.get("case_id"): record for record in records}
    if len(by_id) != len(records) or set(by_id) != set(repository_cases):
        raise ScoringError("raw generations are incomplete, duplicated, or misaligned")
    for case_id, record in by_id.items():
        if (
            record.get("category") != repository_cases[case_id]["category"]
            or _rate(record.get("score"), f"raw.{case_id}.score") != float(per_case[case_id])
            or not isinstance(record.get("validator_detail"), dict)
        ):
            raise ScoringError(
                f"raw generation {case_id} conflicts with protected suite or summary"
            )
    successes = _event_successes(outputs_root)
    event_payload = successes.get(card["run_id"])
    if event_payload is None or event_payload.get("run_card_sha256") != sha256_file(card_path):
        raise ScoringError("evaluation has no successful event-chain record")
    _verify_observed_head(card, outputs_root)
    if baseline:
        expected = policy.seed_model
        model = summary["resolved_model"]
        if (
            model.get("kind") != "hub_model"
            or model.get("base_model")
            != {"repo_id": expected["repo_id"], "revision": expected["revision"]}
            or model.get("artifact") is not None
            or model.get("tokenizer_sha256") != expected["tokenizer_sha256"]
            or model.get("chat_template_sha256") != expected["chat_template_sha256"]
            or model.get("model_config_sha256") != expected["model_config_sha256"]
            or model.get("architecture") != expected["architecture"]
        ):
            raise ScoringError("baseline is not the exact protected seed model")
    else:
        source_run_id = summary["resolved_model"].get("source_run_id")
        if not isinstance(source_run_id, str):
            raise ScoringError("candidate evaluation has no exact source run")
        resolved = resolve_candidate(source_run_id, policy, outputs_root=outputs_root)
        if resolved.to_dict() != summary["resolved_model"]:
            raise ScoringError(
                "candidate evaluation did not use the current verified candidate bytes"
            )
        expected_inputs = [resolved.artifact] if resolved.artifact is not None else []
        if card["inputs"] != expected_inputs:
            raise ScoringError("eval run-card model input differs from candidate checkpoint")
        candidate_card = _load_json(outputs_root / "runs" / source_run_id / "run_card.json")
        data = candidate_card.get("data")
        if (
            not isinstance(data, dict)
            or data.get("license_status") != "usable"
            or data.get("leakage_statistics") != {"status": "clean", "hit_count": 0}
        ):
            raise ScoringError(
                "candidate training data lacks clean leakage and usable license evidence"
            )
    return VerifiedEvaluation(run_dir, summary_path, card_path, summary, card, records, summary_ref)


def load_baseline(
    baseline_root: Path,
    policy: Policy,
    *,
    outputs_root: Path,
    repository_root: Path,
) -> VerifiedEvaluation:
    pointer_path = baseline_root / "current.json"
    pointer = _load_json(pointer_path)
    if (
        set(pointer) != {"schema_version", "run_id", "summary_sha256", "run_card_sha256"}
        or pointer.get("schema_version") != 1
    ):
        raise ScoringError("baseline pointer schema is invalid")
    run_dir = baseline_root / str(pointer["run_id"])
    if (
        sha256_file(run_dir / "summary.json") != pointer["summary_sha256"]
        or sha256_file(run_dir / "run_card.json") != pointer["run_card_sha256"]
    ):
        raise ScoringError("baseline pointer hashes do not match immutable artifacts")
    return verify_evaluation(
        run_dir,
        policy,
        outputs_root=outputs_root,
        baseline=True,
        repository_root=repository_root,
    )


def paired_bootstrap(
    candidate: list[float], baseline: list[float], *, samples: int, seed: int, confidence: float
) -> dict[str, Any]:
    if not candidate or len(candidate) != len(baseline):
        raise ScoringError("paired bootstrap inputs are empty or misaligned")
    deltas = [run - base for run, base in zip(candidate, baseline, strict=True)]
    effect = sum(deltas) / len(deltas)
    random_generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(deltas[random_generator.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    estimates.sort()
    tail = (1.0 - confidence) / 2
    low = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return {"effect": effect, "ci95": [low, high], "n": len(deltas)}


def _vectors(evaluation: VerifiedEvaluation) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {
        key: []
        for key in (
            "german_instruction",
            "format_following",
            "reasoning_core",
            "longcontext",
            "german_language_retention",
            "english_bleed_rate",
            "empty_output_rate",
            "refusal_rate",
            "over_refusal_rate",
            "safety",
        )
    }
    for record in evaluation.raw_records:
        category = record["category"]
        score = float(record["score"])
        detail = record["validator_detail"]
        if category in vectors:
            vectors[category].append(score)
        if category == "german_language_retention":
            vectors["english_bleed_rate"].append(float(bool(detail.get("english_bleed"))))
        if category == "over_refusal":
            vectors["over_refusal_rate"].append(1.0 - score)
        vectors["empty_output_rate"].append(float(bool(detail.get("empty"))))
        vectors["refusal_rate"].append(float(bool(detail.get("refusal"))))
    return vectors


def score_evaluations(
    candidate: VerifiedEvaluation, baseline: VerifiedEvaluation, policy: Policy
) -> dict[str, Any]:
    if candidate.summary["suite_hash"] != baseline.summary["suite_hash"]:
        raise ScoringError("candidate and baseline suite hashes differ")
    if candidate.summary["policy_sha256"] != baseline.summary["policy_sha256"]:
        raise ScoringError("candidate and baseline policy hashes differ")
    if set(candidate.summary["per_case"]) != set(baseline.summary["per_case"]):
        raise ScoringError("candidate and baseline cases are not paired")
    candidate_metrics, baseline_metrics = candidate.summary["metrics"], baseline.summary["metrics"]
    scalar_names = [
        name for name in policy.document["evaluation"]["required_metrics"] if name != "lm_eval"
    ]
    deltas = {
        name: float(candidate_metrics[name]) - float(baseline_metrics[name])
        for name in scalar_names
    }
    lm_deltas = {
        task: float(candidate_metrics["lm_eval"][task]) - float(baseline_metrics["lm_eval"][task])
        for task in policy.document["evaluation"]["lm_eval_tasks"]
    }
    bootstrap = policy.document["evaluation"]["bootstrap"]
    candidate_vectors, baseline_vectors = _vectors(candidate), _vectors(baseline)
    statistics = {
        name: paired_bootstrap(
            candidate_vectors[name],
            baseline_vectors[name],
            samples=int(bootstrap["samples"]),
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap["confidence"]),
        )
        for name in candidate_vectors
    }
    promotion = policy.document["scoring"]["promotion"]
    weights = policy.document["scoring"]["weights"]
    max_lm_regression = max(
        0.0,
        max((-value for value in lm_deltas.values()), default=0.0)
        - promotion["lm_eval_regression_tolerance"],
    )
    english_penalty = max(
        0.0, float(candidate_metrics["english_bleed_rate"]) - promotion["english_bleed_max"]
    )
    response_penalty = max(
        0.0, float(candidate_metrics["empty_output_rate"]) - promotion["empty_output_max"]
    ) + max(0.0, deltas["refusal_rate"] - promotion["refusal_spike_max"])
    safety_penalty = max(0.0, -deltas["safety"]) + max(0.0, deltas["over_refusal_rate"])
    score = (
        weights["german_instruction"] * deltas["german_instruction"]
        + weights["format_following"] * deltas["format_following"]
        + weights["reasoning_core"] * deltas["reasoning_core"]
        + weights["longcontext"] * deltas["longcontext"]
        + weights["lm_eval_regression"] * max_lm_regression
        + weights["english_bleed"] * english_penalty
        + weights["response_suppression"] * response_penalty
        + weights["safety_regression"] * safety_penalty
    )
    gates = {
        "positive_score": score > promotion["min_score"],
        "german_instruction": deltas["german_instruction"]
        >= promotion["german_instruction_min_delta"],
        "format_following": deltas["format_following"] >= promotion["format_following_min_delta"],
        "reasoning_core": deltas["reasoning_core"] >= promotion["reasoning_core_min_delta"],
        "longcontext": deltas["longcontext"] >= promotion["longcontext_min_delta"],
        "safety": deltas["safety"] >= promotion["safety_min_delta"],
        "lm_eval": all(
            value >= -promotion["lm_eval_regression_tolerance"] for value in lm_deltas.values()
        ),
        "english_bleed": candidate_metrics["english_bleed_rate"] <= promotion["english_bleed_max"],
        "empty_output": candidate_metrics["empty_output_rate"] <= promotion["empty_output_max"],
        "refusal": deltas["refusal_rate"] <= promotion["refusal_spike_max"],
        "over_refusal": deltas["over_refusal_rate"] <= promotion["over_refusal_spike_max"],
    }
    return {
        "status": "passed" if all(gates.values()) else "rejected",
        "score": score,
        "deltas": deltas,
        "lm_eval_deltas": lm_deltas,
        "penalties": {
            "lm_eval_regression": max_lm_regression,
            "english_bleed": english_penalty,
            "response_suppression": response_penalty,
            "safety_regression": safety_penalty,
        },
        "gates": gates,
        "statistics": statistics,
    }


def _published_ref(source: Path, destination: Path, repository_root: Path) -> ArtifactRef:
    measured = ArtifactRef.from_path(source, role="score_artifact", media_type="application/json")
    try:
        stored = destination.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        stored = str(destination.resolve())
    return ArtifactRef(
        stored,
        measured.kind,
        measured.role,
        measured.sha256,
        measured.size_bytes,
        measured.media_type,
    )


def create_score(
    eval_run_id: str,
    *,
    policy: Policy,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    started = time.monotonic()
    score_run_id = new_run_id("score")
    events = EventLog(outputs_root)
    start_head = events.append("run_started", score_run_id, {"run_type": "score"})
    baseline = load_baseline(
        outputs_root / "baseline",
        policy,
        outputs_root=outputs_root,
        repository_root=repository_root,
    )
    candidate = verify_evaluation(
        outputs_root / "evals" / eval_run_id,
        policy,
        outputs_root=outputs_root,
        baseline=False,
        repository_root=repository_root,
    )
    result = score_evaluations(candidate, baseline, policy)
    candidate_run_id = candidate.summary["resolved_model"]["source_run_id"]
    staging = outputs_root / "scores/.staging" / score_run_id
    final = outputs_root / "scores" / score_run_id
    run_staging = outputs_root / "runs/.staging" / score_run_id
    run_final = outputs_root / "runs" / score_run_id
    staging.mkdir(parents=True)
    run_staging.mkdir(parents=True)
    artifact = {
        "schema_version": 1,
        "run_id": score_run_id,
        "mode": "real",
        "status": result["status"],
        "candidate_run_id": candidate_run_id,
        "candidate_eval_run_id": candidate.run_card["run_id"],
        "baseline_eval_run_id": baseline.run_card["run_id"],
        "policy_sha256": sha256_file(policy.path),
        "suite_hash": suite_hash(),
        "candidate_summary_sha256": sha256_file(candidate.summary_path),
        "baseline_summary_sha256": sha256_file(baseline.summary_path),
        "candidate_checkpoint": candidate.summary["resolved_model"]["artifact"],
        "score": result["score"],
        "deltas": result["deltas"],
        "lm_eval_deltas": result["lm_eval_deltas"],
        "penalties": result["penalties"],
        "gates": result["gates"],
        "statistics": result["statistics"],
        "event_head_observed": {
            key: start_head[key] for key in ("sequence", "last_event_hash", "log_sha256")
        },
    }
    atomic_write_json(staging / "score.json", artifact)
    score_ref = _published_ref(staging / "score.json", final / "score.json", repository_root)
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    card = {
        "schema_version": 1,
        "run_id": score_run_id,
        "run_type": "score",
        "mode": "real",
        "status": "succeeded" if result["status"] == "passed" else "rejected",
        "started_at": start_head["event"]["timestamp"],
        "finished_at": now,
        "duration_seconds": time.monotonic() - started,
        "command": ["python", "-m", "boldt_posttrain.cli", "score", "--candidate", eval_run_id],
        "git": provenance.collect_git("HEAD", root=repository_root),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "none",
            "sha256": sha256_bytes(b"null"),
            "resolved_sha256": sha256_bytes(b"null"),
        },
        "inputs": [
            candidate.summary_ref.to_dict(),
            baseline.summary_ref.to_dict(),
            candidate.summary["resolved_model"]["artifact"],
        ],
        "outputs": [score_ref.to_dict()],
        "model": candidate.summary["resolved_model"],
        "data": {"suite_hash": suite_hash()},
        "parameters": policy.document["scoring"],
        "hardware": provenance.collect_hardware(),
        "environment": {
            **provenance.collect_environment(),
            "event_head": artifact["event_head_observed"],
        },
        "parents": [candidate.run_card["run_id"], candidate_run_id, baseline.run_card["run_id"]],
        "compatibility_fingerprint": sha256_bytes(
            canonical_json_bytes({"policy": sha256_file(policy.path), "suite": suite_hash()})
        ),
        "error": None,
    }
    validate_run_card(card)
    atomic_write_json(run_staging / "run_card.json", card)
    os.replace(staging, final)
    os.replace(run_staging, run_final)
    finish_status = "succeeded" if result["status"] == "passed" else "rejected"
    events.append(
        "run_finished",
        score_run_id,
        {
            "status": finish_status,
            "run_card_sha256": sha256_file(run_final / "run_card.json"),
        },
    )
    pointer = {
        "schema_version": 1,
        "candidate_run_id": candidate_run_id,
        "score_run_id": score_run_id,
        "score_sha256": sha256_file(final / "score.json"),
        "run_card_sha256": sha256_file(run_final / "run_card.json"),
    }
    with exclusive_lock(outputs_root / "scores/.scores.lock"):
        atomic_write_json(
            outputs_root / "scores/by_candidate" / f"{candidate_run_id}.json", pointer
        )
    return {
        "status": result["status"],
        "score_run_id": score_run_id,
        "candidate_run_id": candidate_run_id,
        "score": result["score"],
    }


def load_candidate_score(
    candidate_run_id: str,
    policy: Policy,
    *,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = _load_json(outputs_root / "scores/by_candidate" / f"{candidate_run_id}.json")
    if (
        set(pointer)
        != {"schema_version", "candidate_run_id", "score_run_id", "score_sha256", "run_card_sha256"}
        or pointer.get("schema_version") != 1
        or pointer.get("candidate_run_id") != candidate_run_id
    ):
        raise ScoringError("candidate score pointer schema or identity is invalid")
    score_path = outputs_root / "scores" / pointer["score_run_id"] / "score.json"
    card_path = outputs_root / "runs" / pointer["score_run_id"] / "run_card.json"
    if (
        sha256_file(score_path) != pointer["score_sha256"]
        or sha256_file(card_path) != pointer["run_card_sha256"]
    ):
        raise ScoringError("candidate score pointer hashes do not match")
    artifact, card = _load_json(score_path), _load_json(card_path)
    validate_run_card(card)
    if (
        artifact.get("candidate_run_id") != candidate_run_id
        or card["run_type"] != "score"
        or card["mode"] != "real"
    ):
        raise ScoringError("score artifact or run card identity is invalid")
    event_payload = _event_successes(outputs_root).get(card["run_id"])
    if event_payload is None or event_payload.get("run_card_sha256") != sha256_file(card_path):
        raise ScoringError("score run card is not anchored by a successful event")
    _verify_observed_head(card, outputs_root)
    if artifact.get("event_head_observed") != card["environment"].get("event_head"):
        raise ScoringError("score artifact and run card observed different event heads")
    score_ref = _card_output(card, "score_artifact")
    if verify_artifact_ref(score_ref, root=repository_root) != score_path.resolve():
        raise ScoringError("score run card does not hash its score artifact")
    if (
        artifact.get("policy_sha256") != sha256_file(policy.path)
        or artifact.get("suite_hash") != suite_hash()
    ):
        raise ScoringError("score policy or suite hash is stale")
    EventLog(outputs_root).validate()
    resolve_candidate(candidate_run_id, policy, outputs_root=outputs_root)
    baseline = load_baseline(
        outputs_root / "baseline",
        policy,
        outputs_root=outputs_root,
        repository_root=repository_root,
    )
    candidate = verify_evaluation(
        outputs_root / "evals" / artifact["candidate_eval_run_id"],
        policy,
        outputs_root=outputs_root,
        baseline=False,
        repository_root=repository_root,
    )
    if (
        artifact.get("candidate_summary_sha256") != sha256_file(candidate.summary_path)
        or artifact.get("baseline_summary_sha256") != sha256_file(baseline.summary_path)
        or artifact.get("candidate_checkpoint") != candidate.summary["resolved_model"]["artifact"]
    ):
        raise ScoringError("score artifact input hashes or checkpoint reference are stale")
    recomputed = score_evaluations(candidate, baseline, policy)
    for key in ("status", "score", "deltas", "lm_eval_deltas", "penalties", "gates", "statistics"):
        if artifact[key] != recomputed[key]:
            raise ScoringError(f"stored score field {key} differs from deterministic recomputation")
    return artifact, card


def run_cli(args) -> tuple[dict[str, Any], int]:
    result = create_score(args.candidate, policy=load_policy())
    return result, 0 if result["status"] == "passed" else 1
