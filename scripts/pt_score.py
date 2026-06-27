#!/usr/bin/env python3
"""Deterministic scoring of a German post-training candidate against a baseline (pure stdlib).

PROTECTED SURFACE — thin CLI over ``boldt_posttrain.scoring.score_run`` (the one definition of the
score and the fail-closed gates). It claims no metrics: it only compares two saved eval summaries.
Dry-run candidates can NEVER pass (their metrics are unmeasured plumbing).

    python scripts/pt_score.py --config configs/posttrain/current.json --candidate my-label \
        --baseline outputs/posttrain/baseline/summary.json --out outputs/posttrain/score-latest.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain import scoring  # noqa: E402

DEFAULT_BASELINE = ROOT / "outputs" / "posttrain" / "baseline" / "summary.json"
EVALS = ROOT / "outputs" / "posttrain" / "evals"


def _resolve_run(run: Optional[str], candidate: Optional[str]) -> pathlib.Path:
    if run:
        return pathlib.Path(run)
    if candidate:
        return EVALS / candidate / "summary.json"
    raise SystemExit("error: pass --run <summary.json> or --candidate <label>")


def _to_markdown(result, run_path, baseline_path) -> str:
    lines = [f"# Post-training score: **{result['status']}**", "",
             f"- score: `{result['score']}`",
             f"- run: `{run_path}`", f"- baseline: `{baseline_path}`", "",
             "| delta | value |", "|---|---:|"]
    for k, v in result["deltas"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "| penalty | value |", "|---|---:|"]
    for k, v in result["penalties"].items():
        lines.append(f"| {k} | {v} |")
    if result["lm_eval_deltas"]:
        lines += ["", "| lm-eval task | Δ (run-base) |", "|---|---:|"]
        for k, v in result["lm_eval_deltas"].items():
            lines.append(f"| {k} | {v} |")
    if result["failed_gates"]:
        lines += ["", "## Failed gates"]
        for g in result["failed_gates"]:
            lines.append(f"- **{g['name']}**: value `{g['value']}` vs threshold `{g['threshold']}`")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--run", default=None, help="path to a candidate eval summary.json")
    ap.add_argument("--candidate", default=None, help="label under outputs/posttrain/evals/")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    # accepted for command-template compatibility; scoring is deterministic regardless of mode.
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args(argv)

    run_path = _resolve_run(args.run, args.candidate)
    baseline_path = pathlib.Path(args.baseline)
    if not run_path.exists():
        print(json.dumps({"status": "fail", "error": f"run summary not found: {run_path}"},
                         ensure_ascii=False))
        return 2
    if not baseline_path.exists():
        print(json.dumps({"status": "fail", "error": f"baseline summary not found: {baseline_path} "
                          "— run /pt-baseline first", "failed_gates": ["baseline_missing"]},
                         ensure_ascii=False))
        return 2

    run = json.loads(run_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    try:
        cfg = cfgmod.resolve_config(pathlib.Path(args.config))
        thresholds = scoring.thresholds_from_config(cfg)
    except Exception:
        thresholds = {}

    result = scoring.score_run(run, baseline, thresholds)
    result["inputs"] = {"run": str(run_path), "baseline": str(baseline_path)}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.format == "markdown":
        print(_to_markdown(result, run_path, baseline_path))
    else:
        print(json.dumps({"status": result["status"], "score": result["score"],
                          "failed_gates": [g["name"] for g in result["failed_gates"]]},
                         ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
