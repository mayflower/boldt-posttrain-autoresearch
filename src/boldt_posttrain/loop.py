"""One deterministic, globally budgeted post-training experiment step."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from . import config as config_module
from .artifacts import RUN_ID_RE, EventLog, atomic_write_json, new_run_id, sha256_file
from .data_pipeline import verify_data_manifest
from .distillation import _teacher_license, distill_and_train, extract_prompts
from .evaluation import _publish_evaluation
from .frontier import (
    _integrity_check,
    current_frontier_hash,
    frontier_status,
    promote_candidate,
)
from .merge import run_search
from .policy import Policy, load_policy
from .preference import _manifest_rows, train_preference_adapter
from .resolver import OUTPUTS, resolve_model
from .scoring import create_score, load_baseline
from .training import load_manifest_rows, train_adapter

ROOT = Path(__file__).resolve().parents[2]


class LoopError(RuntimeError):
    """A loop stage failed technically or violated fresh-artifact sequencing."""


def _remaining(deadline: float, *, stage: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LoopError(f"global budget exhausted before {stage}")
    return remaining


def _execute_lever(
    config: config_module.ExperimentConfig,
    policy: Policy,
    manifest: Mapping[str, Any],
    *,
    deadline: float,
    outputs_root: Path,
    repository_root: Path,
    allow_checkpoints: bool,
    allow_gpu: bool,
) -> dict[str, Any]:
    lever = config.document["experiment"]["lever"]
    training = config.document["training"]
    budget = _remaining(deadline, stage=lever) / 60
    if lever in {"sft", "sft-specialist", "cpt", "cpt-specialist"}:
        kind = "cpt" if lever.startswith("cpt") else "sft"
        dataset = load_manifest_rows(manifest, kind, root=repository_root)
        return train_adapter(
            kind=kind,
            model_source=policy.seed_model["repo_id"],
            revision=policy.seed_model["revision"],
            dataset=dataset,
            output_root=outputs_root / "checkpoints",
            policy=policy,
            experiment=training,
            target_modules=training["target_modules"],
            device="cuda:0",
            qlora=training["method"] == "qlora",
            allow_checkpoints=allow_checkpoints,
            budget_minutes=budget,
            repository_root=repository_root,
            data_metadata=manifest,
        )
    if lever in {"preference", "pref-specialist"}:
        preference = config.document["preference"]
        return train_preference_adapter(
            method=preference["method"],
            model_source=policy.seed_model["repo_id"],
            revision=policy.seed_model["revision"],
            rows=_manifest_rows(manifest),
            output_root=outputs_root / "checkpoints",
            policy=policy,
            training=training,
            preference=preference,
            target_modules=training["target_modules"],
            device="cuda:0",
            qlora=training["method"] == "qlora",
            allow_checkpoints=allow_checkpoints,
            budget_minutes=budget,
            repository_root=repository_root,
            data_metadata=manifest,
        )
    if lever == "distill":
        settings = config.document["distillation"]
        teacher_name = settings["teacher"]
        if RUN_ID_RE.fullmatch(teacher_name):
            teacher = resolve_model(
                policy=policy,
                candidate=teacher_name,
                outputs_root=outputs_root,
            )
        else:
            teacher = resolve_model(policy=policy, model=teacher_name)
        license_id = _teacher_license(teacher, policy, settings["teacher_license"])
        return distill_and_train(
            teacher=teacher,
            teacher_license=license_id,
            student_model_source=policy.seed_model["repo_id"],
            student_model_revision=policy.seed_model["revision"],
            prompts=extract_prompts(
                manifest,
                repository_root=repository_root,
                maximum=settings["max_prompts"],
            ),
            output_data_root=outputs_root / "data",
            output_checkpoint_root=outputs_root / "checkpoints",
            policy=policy,
            training=training,
            generation=settings,
            target_modules=training["target_modules"],
            device="cuda:0",
            qlora=training["method"] == "qlora",
            allow_checkpoints=allow_checkpoints,
            budget_minutes=budget,
            repository_root=repository_root,
        )
    if lever == "merge":
        settings = config.document["merge"]
        result = run_search(
            candidate_ids=settings["inputs"],
            methods=settings["methods"],
            parameters=settings["parameters"],
            dtype=settings["dtype"],
            policy=policy,
            allow_checkpoints=allow_checkpoints,
            allow_gpu=allow_gpu,
            budget_minutes=budget,
            outputs_root=outputs_root,
            repository_root=repository_root,
        )
        if len(result["candidates"]) != 1:
            raise LoopError("one loop round must produce exactly one merge candidate")
        return result["candidates"][0]
    raise LoopError(
        "loop experiment lever must produce one candidate: sft, cpt, preference, distill, or merge"
    )


def run_experiment(
    *,
    config_path: Path,
    base_ref: str,
    budget_minutes: float,
    promote: bool,
    allow_checkpoints: bool,
    allow_gpu: bool,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> tuple[dict[str, Any], int]:
    loop_id = new_run_id("loop")
    loop_dir = outputs_root / "loops" / loop_id
    started = time.monotonic()
    deadline = started + budget_minutes * 60
    events = EventLog(outputs_root)
    start = events.append("run_started", loop_id, {"run_type": "loop"})
    verdict: dict[str, Any] = {
        "schema_version": 1,
        "loop_id": loop_id,
        "status": "running",
        "disposition": None,
        "started_at": start["event"]["timestamp"],
        "base_ref": base_ref,
        "budget_seconds": budget_minutes * 60,
        "stages": {},
        "error": None,
    }
    exit_code = 4
    try:
        policy = load_policy()
        config = config_module.load_experiment(config_path)
        from . import provenance

        provenance.resolve_base_ref(base_ref, root=repository_root)
        load_baseline(
            outputs_root / "baseline",
            policy,
            outputs_root=outputs_root,
            repository_root=repository_root,
        )
        manifest = verify_data_manifest(
            outputs_root / "data",
            policy,
            repository_root=repository_root,
        )
        previous_runs = {
            path.parent.name for path in (outputs_root / "runs").glob("*/run_card.json")
        }
        lever_result = _execute_lever(
            config,
            policy,
            manifest,
            deadline=deadline,
            outputs_root=outputs_root,
            repository_root=repository_root,
            allow_checkpoints=allow_checkpoints,
            allow_gpu=allow_gpu,
        )
        candidate_run_id = lever_result.get("student_run_id") or lever_result.get("run_id")
        if not isinstance(candidate_run_id, str) or candidate_run_id in previous_runs:
            raise LoopError("lever did not produce exactly one fresh candidate run ID")
        if lever_result.get("status") == "budget_exhausted":
            verdict.update(status="budget_exhausted", disposition="rejected")
            verdict["stages"]["lever"] = lever_result
            exit_code = 1
        else:
            resolved = resolve_model(
                policy=policy,
                candidate=candidate_run_id,
                outputs_root=outputs_root,
            )
            verdict["stages"]["lever"] = lever_result
            eval_result = _publish_evaluation(
                resolved=resolved,
                policy=policy,
                config_path=config_path,
                output_root=outputs_root / "evals",
                baseline=False,
                replace_baseline=False,
                device="cuda:0",
                repository_root=repository_root,
                deadline=deadline,
            )
            verdict["stages"]["evaluation"] = eval_result
            previous_scores = {
                path.parent.name for path in (outputs_root / "scores").glob("*/score.json")
            }
            _remaining(deadline, stage="scoring")
            score_result = create_score(
                eval_result["run_id"],
                policy=policy,
                outputs_root=outputs_root,
                repository_root=repository_root,
            )
            if score_result["score_run_id"] in previous_scores:
                raise LoopError("loop attempted to reuse a previous score artifact")
            verdict["stages"]["scoring"] = score_result
            verdict["candidate_run_id"] = candidate_run_id
            verdict["candidate_eval_run_id"] = eval_result["run_id"]
            verdict["score_run_id"] = score_result["score_run_id"]
            verdict["disposition"] = (
                "candidate" if score_result["status"] == "passed" else "rejected"
            )
            integrity = _integrity_check(base_ref, repository_root, policy)
            verdict["stages"]["integrity"] = integrity
            if score_result["status"] == "passed" and promote:
                promotion = promote_candidate(
                    candidate_run_id,
                    base_ref=base_ref,
                    expected_current_sha256=current_frontier_hash(outputs_root / "frontier"),
                    policy=policy,
                    outputs_root=outputs_root,
                    repository_root=repository_root,
                )
                verdict["stages"]["promotion"] = promotion
                verdict.update(status="promoted", disposition="promoted")
                exit_code = 0
            elif score_result["status"] == "passed":
                verdict["status"] = "succeeded"
                exit_code = 0
            else:
                verdict["status"] = "rejected"
                exit_code = 1
    except Exception as exc:
        verdict.update(status="failed", disposition=None, error=f"{type(exc).__name__}: {exc}")
        exit_code = 4
    verdict["duration_seconds"] = time.monotonic() - started
    verdict["remaining_seconds"] = max(0.0, deadline - time.monotonic())
    verdict["finished_at"] = (
        __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    )
    atomic_write_json(loop_dir / "verdict.json", verdict)
    events.append(
        "run_finished",
        loop_id,
        {
            "status": verdict["status"],
            "verdict_sha256": sha256_file(loop_dir / "verdict.json"),
        },
    )
    return verdict, exit_code


def verified_status(
    *,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    policy = load_policy()
    event_log = EventLog(outputs_root)
    events: list[dict[str, Any]] = []
    if event_log.log_path.exists() or event_log.head_path.exists():
        event_log.validate()
        events = [json.loads(line) for line in event_log.log_path.read_text().splitlines()]
    anchored_cards = {
        (event["run_id"], event["payload"].get("run_card_sha256"))
        for event in events
        if event["event_type"] in {"run_finished", "candidate_promoted"}
    }
    records: list[dict[str, Any]] = []
    unverified: list[str] = []
    for card_path in sorted((outputs_root / "runs").glob("*/run_card.json")):
        try:
            card = json.loads(card_path.read_text())
            from .artifacts import validate_run_card

            validate_run_card(card)
            if (card["run_id"], sha256_file(card_path)) not in anchored_cards:
                raise LoopError("run card has no matching event anchor")
            records.append(
                {
                    "run_id": card["run_id"],
                    "run_type": card["run_type"],
                    "status": card["status"],
                    "verified": True,
                }
            )
        except Exception:
            unverified.append(card_path.parent.name)
    plans = []
    for path in sorted((outputs_root / "plans").glob("*/plan.json")):
        try:
            plan = json.loads(path.read_text())
            if plan.get("schema_version") == 1 and plan.get("plan_id") == path.parent.name:
                plans.append(plan["plan_id"])
            else:
                unverified.append(path.parent.name)
        except (OSError, json.JSONDecodeError):
            unverified.append(path.parent.name)
    loops = []
    event_verdicts = {
        (event["run_id"], event["payload"].get("verdict_sha256"))
        for event in events
        if event["event_type"] == "run_finished"
    }
    for path in sorted((outputs_root / "loops").glob("*/verdict.json")):
        try:
            verdict = json.loads(path.read_text())
            if (verdict.get("loop_id"), sha256_file(path)) not in event_verdicts:
                raise LoopError("loop verdict has no event anchor")
            loops.append({"loop_id": verdict["loop_id"], "status": verdict["status"]})
        except Exception:
            unverified.append(path.parent.name)
    return {
        "status": "succeeded",
        "plans": plans,
        "runs": records,
        "loops": loops,
        "frontier": frontier_status(
            policy=policy,
            outputs_root=outputs_root,
            repository_root=repository_root,
        )["frontier"],
        "legacy_or_unverified": sorted(set(unverified)),
    }


def run_cli(args) -> tuple[dict[str, Any], int]:
    if args.command in {"status", "report"}:
        return verified_status(), 0
    return run_experiment(
        config_path=ROOT / args.config,
        base_ref=args.base_ref,
        budget_minutes=args.budget_minutes,
        promote=args.promote,
        allow_checkpoints=args.allow_checkpoints,
        allow_gpu=args.allow_gpu,
    )
