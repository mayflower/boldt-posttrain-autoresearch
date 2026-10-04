"""Human-readable progress lines and round reports built from published artifacts.

Progress goes to stderr so every command's stdout stays one JSON result. Reports are
rendered from the verdict, run card, score and evaluation summaries; they add no facts
of their own and are regenerated with `pt report --loop <loop-id>`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .policy import Policy

# gate -> (metric, compared value, operator, policy.scoring.promotion key)
METRIC_GATES: dict[str, tuple[str, str, str, str]] = {
    "german_instruction": ("german_instruction", "delta", ">=", "german_instruction_min_delta"),
    "format_following": ("format_following", "delta", ">=", "format_following_min_delta"),
    "reasoning_core": ("reasoning_core", "delta", ">=", "reasoning_core_min_delta"),
    "longcontext": ("longcontext", "delta", ">=", "longcontext_min_delta"),
    "safety": ("safety", "delta", ">=", "safety_min_delta"),
    "english_bleed": ("english_bleed_rate", "delta", "<=", "english_bleed_spike_max"),
    "empty_output": ("empty_output_rate", "delta", "<=", "empty_output_spike_max"),
    "refusal": ("refusal_rate", "delta", "<=", "refusal_spike_max"),
    "over_refusal": ("over_refusal_rate", "delta", "<=", "over_refusal_spike_max"),
}
TRAINING_KEYS = (
    "method",
    "specialist",
    "learning_rate",
    "max_steps",
    "num_train_epochs",
    "warmup_ratio",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "context_length",
    "packing",
    "assistant_only_loss",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "quantization",
    "seed",
)


def progress(message: str) -> None:
    print(f"[pt {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def training_summary(training: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={training[key]}" for key in TRAINING_KEYS if key in training)


def experiment_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of an experiment file that decide what one round trains."""
    lever = document["experiment"]["lever"]
    snapshot: dict[str, Any] = {
        "experiment": dict(document["experiment"]),
        "training": dict(document["training"]),
        "data": dict(document["data"]),
    }
    for block, levers in (
        ("preference", {"preference", "pref-specialist"}),
        ("distillation", {"opd", "distill"}),
        ("online", {"grpo", "rlvr", "opd", "sdpo", "sdft", "distill"}),
        ("seqkd", {"seqkd"}),
        ("merge", {"merge"}),
    ):
        if lever in levers and block in document:
            snapshot[block] = document[block]
    return snapshot


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _number(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "–"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _gate_text(
    gate: str, passed: bool | None, promotion: Mapping[str, Any], value: Any = None
) -> str:
    if passed is None:
        return ""
    _metric, compared, operator, key = METRIC_GATES[gate]
    label = "Δ" if compared == "delta" else "value"
    shown = f" {value:+.3f}" if isinstance(value, (int, float)) else ""
    if passed:
        return f"pass ({label}{shown} {operator} {promotion[key]:+g})"
    violated = {">=": "<", "<=": ">"}[operator]
    return f"FAIL ({label}{shown} {violated} {promotion[key]:+g})"


def _gate_value(gate: str, score: Mapping[str, Any], candidate: Mapping[str, Any]) -> Any:
    metric, compared, _operator, _key = METRIC_GATES[gate]
    return score.get("deltas", {}).get(metric) if compared == "delta" else candidate.get(metric)


def score_lines(
    score: Mapping[str, Any], policy: Policy, candidate: Mapping[str, Any] | None = None
) -> list[str]:
    """One line per failed gate, explaining the failure against the policy threshold."""
    promotion = policy.document["scoring"]["promotion"]
    lines = []
    for gate, passed in score.get("gates", {}).items():
        if passed:
            continue
        if gate == "positive_score":
            lines.append(f"score {score['score']:+.3f} is not above {promotion['min_score']:+g}")
        elif gate == "lm_eval":
            worst = min(score.get("lm_eval_deltas", {}).items(), key=lambda item: item[1])
            lines.append(
                f"lm-eval {worst[0]} Δ {worst[1]:+.3f} below "
                f"-{promotion['lm_eval_regression_tolerance']:g}"
            )
        elif gate in METRIC_GATES:
            value = _gate_value(gate, score, candidate or {})
            lines.append(f"{METRIC_GATES[gate][0]}: {_gate_text(gate, False, promotion, value)}")
    return lines


def render_loop_report(verdict: Mapping[str, Any], *, policy: Policy, outputs_root: Path) -> str:
    promotion = policy.document["scoring"]["promotion"]
    out: list[str] = [f"# Round {verdict['loop_id']}", ""]
    out += [
        f"- Status: **{verdict['status']}**"
        + (
            f" ({verdict['disposition']})"
            if verdict.get("disposition") not in {None, verdict["status"]}
            else ""
        ),
        f"- Started: {verdict.get('started_at')}, duration "
        f"{duration(float(verdict.get('duration_seconds') or 0))} of "
        f"{duration(float(verdict.get('budget_seconds') or 0))} budget",
        f"- Base ref: `{verdict.get('base_ref')}`",
    ]
    if verdict.get("error"):
        out += [f"- Error: `{verdict['error']}`"]

    snapshot = verdict.get("experiment")
    if snapshot:
        experiment = snapshot["experiment"]
        out += [
            "",
            "## Experiment",
            "",
            f"- Name: {experiment.get('name')}",
            f"- Lever: `{experiment.get('lever')}`",
            f"- Hypothesis: {experiment.get('hypothesis')}",
            f"- Training: {training_summary(snapshot['training'])}",
        ]
        for source in snapshot.get("data", {}).get("sources", []):
            out.append(
                f"- Data source: `{source['dataset_id']}@{source['revision'][:12]}` "
                f"config `{source['config']}` split `{source['split']}` ({source['schema']})"
            )
        for block in ("preference", "distillation", "online", "seqkd", "merge"):
            if block in snapshot:
                settings = ", ".join(f"{k}={v}" for k, v in snapshot[block].items())
                out.append(f"- {block}: {settings}")

    candidate_id = verdict.get("candidate_run_id")
    card = (
        _load(outputs_root / "runs" / str(candidate_id) / "run_card.json") if candidate_id else None
    )
    if card:
        metrics = card.get("environment", {}).get("metrics", {})
        data = card.get("data", {})
        out += ["", "## Training", "", f"- Run: `{candidate_id}` ({card['run_type']})"]
        if data.get("run_id"):
            out.append(
                f"- Data manifest: `{data['run_id']}`"
                + (
                    f", {data['token_statistics']['count']} rows, "
                    f"{data['token_statistics']['total']} tokens"
                    if isinstance(data.get("token_statistics"), dict)
                    and "count" in data["token_statistics"]
                    else ""
                )
            )
        if data.get("teacher"):
            out.append(
                f"- Teacher: `{data['teacher']['repo_id']}@{data['teacher']['revision'][:12]}`"
            )
        if metrics:
            out.append(
                "- Result: "
                + ", ".join(
                    f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in metrics.items()
                    if key
                    in {
                        "steps_completed",
                        "epochs_completed",
                        "examples_seen",
                        "tokens_seen",
                        "train_loss",
                        "stop_reason",
                    }
                )
            )
            if "wall_clock_seconds" in metrics:
                peak = metrics.get("peak_gpu_memory_bytes", 0) / 2**30
                out.append(
                    f"- Time: {duration(metrics['wall_clock_seconds'])}, peak GPU memory "
                    f"{peak:.1f} GiB, trainable parameters "
                    f"{metrics.get('trainable_parameters', 0):,} of "
                    f"{metrics.get('total_parameters', 0):,}"
                )
        checkpoint = next(
            (ref for ref in card.get("outputs", []) if ref.get("role", "").endswith("checkpoint")),
            None,
        )
        if checkpoint:
            out.append(f"- Checkpoint: `{checkpoint['path']}` sha256 `{checkpoint['sha256']}`")

    score_id = verdict.get("score_run_id")
    score = _load(outputs_root / "scores" / str(score_id) / "score.json") if score_id else None
    if score:
        baseline = _load(outputs_root / "baseline" / score["baseline_eval_run_id"] / "summary.json")
        candidate = _load(outputs_root / "evals" / score["candidate_eval_run_id"] / "summary.json")
        base_metrics = (baseline or {}).get("metrics", {})
        cand_metrics = (candidate or {}).get("metrics", {})
        metric_gates = {metric: gate for gate, (metric, *_rest) in METRIC_GATES.items()}
        out += [
            "",
            "## Evaluation",
            "",
            f"Baseline `{score['baseline_eval_run_id']}` vs candidate "
            f"`{score['candidate_eval_run_id']}`, suite hash `{score['suite_hash'][:16]}`.",
            "",
            "| Metric | Baseline | Candidate | Δ | 95% CI | n | Gate |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
        for metric, delta in score.get("deltas", {}).items():
            stats = score.get("statistics", {}).get(metric, {})
            ci = stats.get("ci95")
            gate = metric_gates.get(metric)
            out.append(
                f"| {metric} | {_number(base_metrics.get(metric))} | "
                f"{_number(cand_metrics.get(metric))} | {_number(delta, signed=True)} | "
                + (f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "–")
                + f" | {stats.get('n', '–')} | "
                + (
                    _gate_text(
                        gate,
                        score["gates"].get(gate),
                        promotion,
                        _gate_value(gate, score, cand_metrics),
                    )
                    if gate
                    else ""
                )
                + " |"
            )
        out += ["", "| lm-eval task | Baseline | Candidate | Δ |", "| --- | ---: | ---: | ---: |"]
        for task, delta in score.get("lm_eval_deltas", {}).items():
            out.append(
                f"| {task} | {_number(base_metrics.get('lm_eval', {}).get(task))} | "
                f"{_number(cand_metrics.get('lm_eval', {}).get(task))} | "
                f"{_number(delta, signed=True)} |"
            )
        failed = score_lines(score, policy, cand_metrics)
        penalties = ", ".join(f"{k}={v:.3f}" for k, v in score.get("penalties", {}).items())
        out += [
            "",
            "## Score and decision",
            "",
            f"- Score: **{score['score']:+.3f}** ({score['status']}); penalties: {penalties}",
            f"- Gates passed: {sum(score['gates'].values())}/{len(score['gates'])}",
        ]
        out += [f"- Failed: {line}" for line in failed] or ["- All gates passed."]

    stages = verdict.get("stages", {})
    if "integrity" in stages:
        integrity = stages["integrity"]
        out += [
            "",
            "## Integrity and promotion",
            "",
            f"- Integrity: {integrity.get('status')}"
            + (f", violations {integrity['violations']}" if integrity.get("violations") else ""),
        ]
        promotion_stage = stages.get("promotion")
        if promotion_stage and promotion_stage.get("status") == "not_promoted":
            out.append(f"- Not promoted: {promotion_stage['reason']}")
        elif promotion_stage:
            out.append(f"- Promoted: {json.dumps(promotion_stage, sort_keys=True)}")
        else:
            out.append("- Not promoted.")
    return "\n".join(out) + "\n"
