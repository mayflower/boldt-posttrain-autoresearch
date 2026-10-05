"""Where the project stands and what to run next, explained for a human.

Imports stay light (no torch or transformers), so the session-start hook can call
`pt guide` within its timeout. Durations come from earlier run cards when they exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .data_pipeline import verify_data_manifest
from .policy import Policy
from .resolver import OUTPUTS, ROOT
from .scoring import load_baseline


def _check(verify: Callable[[], str]) -> dict[str, Any]:
    try:
        return {"status": "ok", "run_id": verify()}
    except Exception as exc:  # noqa: BLE001 -- report, never raise from status
        return {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"}


def readiness(
    policy: Policy, *, outputs_root: Path = OUTPUTS, repository_root: Path = ROOT
) -> dict[str, Any]:
    """Whether the loop prerequisites verify under the current policy, and what is next."""
    data = _check(
        lambda: verify_data_manifest(
            outputs_root / "data", policy, repository_root=repository_root
        )["run_id"]
    )
    baseline = _check(
        lambda: load_baseline(
            outputs_root / "baseline",
            policy,
            outputs_root=outputs_root,
            repository_root=repository_root,
        ).run_card["run_id"]
    )
    next_command = (
        "/pt-data real"
        if data["status"] != "ok"
        else "/pt-baseline real"
        if baseline["status"] != "ok"
        else "/pt-run <rounds> real"
    )
    return {
        "loop_ready": data["status"] == "ok" and baseline["status"] == "ok",
        "data_manifest": data,
        "baseline": baseline,
        "next_command": next_command,
    }


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _minutes(seconds: float | None) -> str:
    if not seconds:
        return "unknown"
    return f"~{max(1, round(seconds / 60))} min"


def _last_duration(directory: Path, run_type: str) -> float | None:
    cards = [
        card
        for card in (_load(path) for path in sorted(directory.glob("*/run_card.json")))
        if card and card.get("run_type") == run_type and card.get("status") == "succeeded"
    ]
    return float(cards[-1]["duration_seconds"]) if cards else None


def _cause(error: str | None) -> str:
    text = (error or "").lower()
    if "pointer is missing" in text or "no such file" in text or "not found" in text:
        return "it does not exist yet"
    if "policy hash" in text:
        return "policy.json changed since it was made, so it no longer verifies"
    if "suite" in text:
        return "the evaluation suite changed since it was made"
    return f"it does not verify ({error})"


def _rounds(outputs_root: Path) -> list[dict[str, Any]]:
    rounds = []
    for path in sorted((outputs_root / "loops").glob("*/verdict.json")):
        verdict = _load(path)
        if not verdict:
            continue
        score = None
        if verdict.get("score_run_id"):
            document = _load(outputs_root / "scores" / verdict["score_run_id"] / "score.json")
            score = document.get("score") if document else None
        rounds.append(
            {
                "loop_id": verdict.get("loop_id"),
                "status": verdict.get("status"),
                "lever": (verdict.get("experiment") or {}).get("experiment", {}).get("lever"),
                "score": score,
                "error": verdict.get("error"),
            }
        )
    return rounds


def render_guide(
    policy: Policy, *, outputs_root: Path = OUTPUTS, repository_root: Path = ROOT
) -> str:
    state = readiness(policy, outputs_root=outputs_root, repository_root=repository_root)
    data, baseline = state["data_manifest"], state["baseline"]
    rounds = _rounds(outputs_root)
    frontier = _load(outputs_root / "frontier" / "current.json")
    prepare_time = _last_duration(outputs_root / "data", "data_prepare")
    baseline_time = _last_duration(outputs_root / "baseline", "baseline")
    seed = policy.seed_model

    def mark(ok: bool) -> str:
        return "[x]" if ok else "[ ]"

    out = [
        "PostTrain AutoResearch: post-training "
        f"{seed['repo_id']}@{seed['revision'][:12]} for German",
        "",
        "Where you are",
        f"  {mark(data['status'] == 'ok')} 1. Training data   "
        + (f"{data['run_id']}" if data["status"] == "ok" else _cause(data.get("error"))),
        f"  {mark(baseline['status'] == 'ok')} 2. Baseline        "
        + (
            f"{baseline['run_id']}" if baseline["status"] == "ok" else _cause(baseline.get("error"))
        ),
        f"  {mark(bool(frontier))} 3. Research rounds "
        + (
            f"{len(rounds)} so far; champion {frontier.get('candidate_run_id')}"
            f" (score {frontier.get('score', 0):+.3f})"
            if frontier
            else f"{len(rounds)} so far, no champion yet"
        ),
        "",
    ]
    if data["status"] != "ok":
        out += [
            "Next: /pt-data real",
            "  Why: the training data manifest is not usable:",
            f"  {_cause(data.get('error'))}.",
            "  Training and the loop refuse unverified data.",
            "  What it does: lists the allowed Hugging Face datasets, then prepares the source",
            "  in configs/posttrain/secure-current.json (German language filter, dedup,",
            "  benchmark-leakage check) and publishes a hashed manifest. No GPU;",
            f"  prepare took {_minutes(prepare_time)} last time, discovery about 15 min.",
            "  After it: /pt-baseline real, then /pt-run 1 real.",
        ]
    elif baseline["status"] != "ok":
        out += [
            "Next: /pt-baseline real",
            "  Why: the baseline is not usable:",
            f"  {_cause(baseline.get('error'))}.",
            "  Every candidate is scored against it, so the loop cannot run without it.",
            "  What it does: evaluates the unchanged seed model on the protected German suite",
            "  and the lm-eval tasks. An outdated baseline is replaced automatically and kept",
            f"  as history. GPU, took {_minutes(baseline_time)} last time.",
            "  After it: /pt-run 1 real.",
        ]
    else:
        out += [
            "Next: /pt-run 1 real   (one research round; use a larger number for a series)",
            "  What a round does: the agent explains its hypothesis and the settings it changes,",
            "  trains one candidate, evaluates it against the baseline, scores it and promotes it",
            "  only if every gate passes and it beats the champion. Progress lines show training",
            "  step/loss/ETA and evaluation progress; /pt-report <loop-id> explains the result.",
            "  Optional first: /pt-seqkd real (2-3 h GPU) to use Qwen3.8 as teacher (lever seqkd).",
        ]
    if rounds:
        last = rounds[-1]
        out += [
            "",
            f"Last round: {last['loop_id']} {last['status']}"
            + (f", lever {last['lever']}" if last["lever"] else "")
            + (f", score {last['score']:+.3f}" if isinstance(last["score"], (int, float)) else "")
            + (f", error: {last['error']}" if last["error"] else ""),
            f"  Details: /pt-report {last['loop_id']}",
        ]
    out += [
        "",
        "Other commands: /pt-orient (this overview plus policy check), /pt-status (all runs),",
        "/pt-train, /pt-eval, /pt-merge, /pt-promote for single steps.",
        "A nonzero exit is a stop: technical or integrity failures end a series.",
    ]
    return "\n".join(out) + "\n"
