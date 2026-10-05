"""One deterministic, globally budgeted post-training experiment step."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import config as config_module
from .artifacts import RUN_ID_RE, EventLog, atomic_write_json, new_run_id, sha256_file

from .data_pipeline import verify_data_manifest
from .training import load_manifest_rows
from .distillation import _teacher_license
from .evaluation import _publish_evaluation
from .frontier import (
    FrontierNotImproved,
    _integrity_check,
    current_frontier_hash,
    frontier_status,
    promote_candidate,
)
from .merge import run_search
from .online import ONLINE_LEVERS, online_method, validate_online_policy
from .online_training import train_online_candidate
from .policy import Policy, load_policy
from .preference import _manifest_rows, train_preference_adapter
from .resolver import OUTPUTS, resolve_model
from .report import (
    duration,
    experiment_snapshot,
    progress,
    render_loop_report,
    score_lines,
    training_summary,
)
from .guide import readiness
from .scoring import create_score, load_baseline
from .seqkd import validate_seqkd_policy, verify_generation
from .training import train_adapter

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
    if lever == "seqkd":
        settings = validate_seqkd_policy(config.document, policy)
        generated, generation_card = verify_generation(
            settings,
            policy,
            outputs_root=outputs_root,
            repository_root=repository_root,
        )
        return train_adapter(
            kind="sft",
            model_source=policy.seed_model["repo_id"],
            revision=policy.seed_model["revision"],
            dataset=load_manifest_rows(generated, "sft", root=repository_root),
            output_root=outputs_root / "checkpoints",
            policy=policy,
            experiment=training,
            target_modules=training["target_modules"],
            device="cuda:0",
            qlora=training["method"] == "qlora",
            allow_checkpoints=allow_checkpoints,
            budget_minutes=budget,
            repository_root=repository_root,
            input_artifacts=[
                ref for ref in generation_card["outputs"] if ref["role"] == "sft_shard"
            ],
            parent_run_ids=[generation_card["run_id"]],
            lineage={
                "seqkd_generation_run": generation_card["run_id"],
                "teacher": generated["teacher"],
                "prompt_source": generated["prompt_source"],
            },
            data_metadata=generated,
            run_type="train_seqkd",
        )
    if lever in ONLINE_LEVERS:
        validate_online_policy(config.document, policy)
        settings = config.document["distillation"]
        teacher = None
        license_id = None
        if online_method(lever) == "opd":
            teacher_name = settings["teacher"]
            if RUN_ID_RE.fullmatch(teacher_name):
                teacher = resolve_model(
                    policy=policy,
                    candidate=teacher_name,
                    outputs_root=outputs_root,
                )
            else:
                teacher = resolve_model(policy=policy, model=teacher_name)
            if getattr(teacher, "kind", None) == "hub_model" and teacher.base_model == {
                "repo_id": policy.seed_model["repo_id"],
                "revision": policy.seed_model["revision"],
            }:
                raise LoopError(
                    "OPD requires a distinct teacher; the unchanged student seed supplies no additional teaching signal"
                )
            license_id = _teacher_license(teacher, policy, settings["teacher_license"])
        return train_online_candidate(
            config=config.document,
            policy=policy,
            manifest=manifest,
            outputs_root=outputs_root,
            repository_root=repository_root,
            deadline=deadline,
            allow_gpu=allow_gpu,
            allow_checkpoints=allow_checkpoints,
            teacher_ref=teacher,
            teacher_license=license_id,
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
        "loop experiment lever must produce one candidate: sft, cpt, preference, "
        "distill/opd, grpo, rlvr, sdpo, sdft, seqkd, or merge"
    )


_MANUAL_LEVERS = {"sft", "cpt", "preference", "seqkd", *ONLINE_LEVERS}


def train_one_lever(
    *,
    lever: str,
    config_path: Path,
    budget_minutes: float,
    allow_gpu: bool,
    allow_checkpoints: bool,
    specialist: str | None = None,
    preference_method: str | None = None,
    teacher: str | None = None,
    teacher_license: str | None = None,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> tuple[dict[str, Any], int]:
    """Run a single training lever manually and return its resolvable candidate.

    This is the producer the loop uses (`_execute_lever`), so the candidate carries the
    run card and event-chain record `resolver.resolve_candidate` requires. It trains
    only; evaluation, scoring and promotion are separate commands on the returned run id.
    """
    if lever not in _MANUAL_LEVERS:
        return {
            "status": "failed",
            "error": f"manual training lever must be one of {sorted(_MANUAL_LEVERS)}",
        }, 2
    if not (allow_gpu and allow_checkpoints):
        return {
            "status": "failed",
            "error": "manual training requires --real --allow-gpu --allow-checkpoints",
        }, 2
    try:
        policy = load_policy()
        config = config_module.load_experiment(config_path)
        # The command names the lever; the config's own lever is advisory here.
        config.document["experiment"]["lever"] = lever
        if specialist:
            config.document["training"]["specialist"] = specialist
        if preference_method is not None:
            if lever != "preference" or preference_method not in {"dpo", "kto", "orpo"}:
                raise ValueError("preference method requires a preference lever and dpo/kto/orpo")
            config.document["preference"]["method"] = preference_method
        if teacher is not None:
            config.document["distillation"]["teacher"] = teacher
        if teacher_license is not None:
            config.document["distillation"]["teacher_license"] = teacher_license
        if lever in ONLINE_LEVERS:
            validate_online_policy(config.document, policy)
        if lever == "seqkd":
            validate_seqkd_policy(config.document, policy)
        manifest = verify_data_manifest(
            outputs_root / "data", policy, repository_root=repository_root
        )
        progress(
            f"manual {lever} training on data manifest {manifest['run_id']}: "
            f"{training_summary(config.document['training'])}"
        )
        previous_runs = {
            path.parent.name for path in (outputs_root / "runs").glob("*/run_card.json")
        }
        deadline = time.monotonic() + budget_minutes * 60
        result = _execute_lever(
            config,
            policy,
            manifest,
            deadline=deadline,
            outputs_root=outputs_root,
            repository_root=repository_root,
            allow_checkpoints=allow_checkpoints,
            allow_gpu=allow_gpu,
        )
    except Exception as exc:  # noqa: BLE001 -- surface any failure as a nonzero exit
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, 4
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or run_id in previous_runs:
        return {"status": "failed", "error": "lever did not produce one fresh candidate"}, 4
    if result.get("status") != "succeeded":
        return {"status": result.get("status", "failed"), "run_id": run_id}, 4
    resolve_model(policy=policy, candidate=run_id, outputs_root=outputs_root)
    metrics = result.get("metrics", {})
    progress(
        f"manual {lever} training finished: candidate {run_id}"
        + (
            f", {metrics.get('steps_completed')} steps, loss {metrics.get('train_loss', 0):.4f}"
            if metrics
            else ""
        )
    )
    return {"status": "succeeded", "run_id": run_id, "candidate": run_id}, 0


# Evaluation reuses the baseline's suite on a same-sized model; adapter inference measured
# about 1.25x the baseline's duration, so reserve 1.5x plus scoring/integrity time.
EVALUATION_RESERVE_FACTOR = 1.5
EVALUATION_RESERVE_FIXED_SECONDS = 120.0
MINIMUM_TRAINING_SECONDS = 600.0


def evaluation_reserve_seconds(baseline_duration_seconds: float) -> float:
    return (
        EVALUATION_RESERVE_FACTOR * float(baseline_duration_seconds)
        + EVALUATION_RESERVE_FIXED_SECONDS
    )


def _note(build: Callable[[], str | list[str]]) -> None:
    """Emit progress built lazily; a reporting failure never changes a verdict."""
    try:
        message = build()
    except Exception as exc:  # noqa: BLE001
        message = f"(progress unavailable: {type(exc).__name__}: {exc})"
    for line in [message] if isinstance(message, str) else message:
        progress(line)


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
        try:
            verdict["experiment"] = experiment_snapshot(config.document)
        except (KeyError, TypeError):
            pass
        experiment = config.document.get("experiment", {})
        _note(
            lambda: [
                f"round {loop_id}: lever {experiment['lever']} ({experiment['name']}), "
                f"budget {duration(budget_minutes * 60)}",
                f"hypothesis: {experiment['hypothesis']}",
            ]
        )
        from . import provenance

        provenance.resolve_base_ref(base_ref, root=repository_root)
        baseline = load_baseline(
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
        reserve = evaluation_reserve_seconds(baseline.run_card["duration_seconds"])
        training_deadline = deadline - reserve
        if training_deadline - time.monotonic() < MINIMUM_TRAINING_SECONDS:
            raise LoopError(
                f"budget {duration(budget_minutes * 60)} leaves under "
                f"{duration(MINIMUM_TRAINING_SECONDS)} for training after reserving "
                f"{duration(reserve)} for evaluation; raise --budget-minutes"
            )
        _note(
            lambda: [
                f"prerequisites verified: baseline {baseline.run_card['run_id']}, "
                f"data manifest {manifest['run_id']}",
                f"budget: {duration(training_deadline - time.monotonic())} for training, "
                f"{duration(reserve)} reserved for evaluation and scoring "
                f"(baseline evaluation took {duration(baseline.run_card['duration_seconds'])})",
                f"training ({experiment['lever']}): "
                f"{training_summary(config.document['training'])}",
            ]
        )
        previous_runs = {
            path.parent.name for path in (outputs_root / "runs").glob("*/run_card.json")
        }
        lever_result = _execute_lever(
            config,
            policy,
            manifest,
            deadline=training_deadline,
            outputs_root=outputs_root,
            repository_root=repository_root,
            allow_checkpoints=allow_checkpoints,
            allow_gpu=allow_gpu,
        )
        candidate_run_id = lever_result.get("student_run_id") or lever_result.get("run_id")
        if not isinstance(candidate_run_id, str) or candidate_run_id in previous_runs:
            raise LoopError("lever did not produce exactly one fresh candidate run ID")
        if lever_result.get("status") != "succeeded":
            raise LoopError(f"lever failed with status {lever_result.get('status')!r}")
        else:
            resolved = resolve_model(
                policy=policy,
                candidate=candidate_run_id,
                outputs_root=outputs_root,
            )
            verdict["stages"]["lever"] = lever_result
            metrics = lever_result.get("metrics") or {}
            _note(
                lambda: [
                    f"training finished: candidate {candidate_run_id}"
                    + (
                        f", {metrics['steps_completed']} steps, loss "
                        f"{metrics['train_loss']:.4f}, stop {metrics['stop_reason']}, "
                        f"{duration(metrics['wall_clock_seconds'])}"
                        if metrics
                        else ""
                    ),
                    "evaluating candidate on the protected suite, "
                    f"{duration(max(0.0, deadline - time.monotonic()))} of budget left",
                ]
            )
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
            _note(
                lambda: (
                    f"score {score_result['score_run_id']}: "
                    f"{score_result['score']:+.3f} ({score_result['status']})"
                )
            )
            _note(
                lambda: [
                    f"gate failed: {line}"
                    for line in score_lines(
                        json.loads(
                            (
                                outputs_root
                                / "scores"
                                / score_result["score_run_id"]
                                / "score.json"
                            ).read_text()
                        ),
                        policy,
                        json.loads(
                            (
                                outputs_root / "evals" / eval_result["run_id"] / "summary.json"
                            ).read_text()
                        )["metrics"],
                    )
                ]
            )
            verdict["candidate_run_id"] = candidate_run_id
            verdict["candidate_eval_run_id"] = eval_result["run_id"]
            verdict["score_run_id"] = score_result["score_run_id"]
            verdict["disposition"] = (
                "candidate" if score_result["status"] == "passed" else "rejected"
            )
            integrity = _integrity_check(base_ref, repository_root, policy)
            verdict["stages"]["integrity"] = integrity
            _note(lambda: f"integrity: {integrity['status']}")
            if score_result["status"] == "passed" and promote:
                try:
                    promotion = promote_candidate(
                        candidate_run_id,
                        base_ref=base_ref,
                        expected_current_sha256=current_frontier_hash(outputs_root / "frontier"),
                        policy=policy,
                        outputs_root=outputs_root,
                        repository_root=repository_root,
                    )
                except FrontierNotImproved as exc:
                    # A valid result, not a technical failure: the round beat the baseline
                    # but not the champion, so the series continues.
                    reason = str(exc)
                    verdict["stages"]["promotion"] = {"status": "not_promoted", "reason": reason}
                    verdict.update(status="rejected", disposition="not_promoted")
                    _note(lambda: f"not promoted: {reason}")
                    exit_code = 1
                else:
                    verdict["stages"]["promotion"] = promotion
                    verdict.update(status="promoted", disposition="promoted")
                    _note(lambda: f"promoted: {candidate_run_id} is the new champion")
                    exit_code = 0
            elif score_result["status"] == "passed":
                verdict["status"] = "succeeded"
                exit_code = 0
            else:
                verdict["status"] = "rejected"
                exit_code = 1
    except Exception as exc:
        verdict.update(status="failed", disposition=None, error=f"{type(exc).__name__}: {exc}")
        progress(f"round failed: {verdict['error']}")
        exit_code = 4
    verdict["duration_seconds"] = time.monotonic() - started
    verdict["remaining_seconds"] = max(0.0, deadline - time.monotonic())
    verdict["finished_at"] = (
        __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    )
    try:
        report = render_loop_report(verdict, policy=load_policy(), outputs_root=outputs_root)
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / "report.md").write_text(report, encoding="utf-8")
        verdict["report"] = str(loop_dir / "report.md")
    except Exception as exc:  # noqa: BLE001 -- a report failure never changes the verdict
        progress(f"report rendering failed: {type(exc).__name__}: {exc}")
    progress(
        f"round {verdict['status']} after {duration(verdict['duration_seconds'])}"
        + (f"; report {verdict['report']}" if verdict.get("report") else "")
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
        "readiness": readiness(policy, outputs_root=outputs_root, repository_root=repository_root),
    }
