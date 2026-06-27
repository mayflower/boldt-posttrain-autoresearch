#!/usr/bin/env python3
"""Promotion gate for a German post-training candidate (pure stdlib).

Does NOT re-implement or weaken the gate. It reuses the protected scorer
(``boldt_posttrain.scoring``) and the protected integrity guard, and additionally requires a
STRICTLY positive German-instruction improvement over the baseline AND the current promoted
frontier. Fail-closed: a missing candidate/baseline summary, a failing score gate, a regression,
or any protected-surface edit blocks promotion. On pass it writes ``outputs/posttrain/frontier.json``
with METADATA POINTERS ONLY — never weights, never a Hugging Face push.

    python scripts/pt_promote.py --candidate my-merge --format markdown
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain import frontier as fr  # noqa: E402
from boldt_posttrain import provenance as prov  # noqa: E402
from boldt_posttrain import scoring  # noqa: E402

OUT = ROOT / "outputs" / "posttrain"
DEFAULT_BASELINE = OUT / "baseline" / "summary.json"
FRONTIER = OUT / "frontier.json"


def _integrity(base_ref: Optional[str]) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "check_posttrain_integrity", ROOT / "scripts" / "check_posttrain_integrity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate(mod.changed_paths(base_ref))


def evaluate_promotion(candidate: str, cfg: Dict[str, Any], baseline_path: pathlib.Path,
                       base_ref: Optional[str]) -> Dict[str, Any]:
    cand_summary = fr.EVALS / candidate / "summary.json"
    failed: List[str] = []
    if not cand_summary.exists():
        return {"candidate": candidate, "promotable": False,
                "failed_gates": ["candidate_summary_missing"],
                "error": f"no eval summary at {cand_summary} — run /pt-eval first"}
    if not baseline_path.exists():
        return {"candidate": candidate, "promotable": False,
                "failed_gates": ["baseline_missing"],
                "error": f"no baseline at {baseline_path} — run /pt-baseline first"}

    run = json.loads(cand_summary.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    score = scoring.score_run(run, baseline, scoring.thresholds_from_config(cfg))
    if score["status"] != "pass":
        failed += [g["name"] for g in score["failed_gates"]]

    # Promotion is STRICTER than the score gate: require real positive headline improvement.
    d_gi = score["deltas"].get("german_instruction")
    if not (isinstance(d_gi, (int, float)) and d_gi > 0):
        failed.append("german_instruction_not_improved")

    # Must also beat the current promoted frontier's aggregate (if one exists).
    cur = fr.current_frontier()
    cand_agg = fr.aggregate(run.get("metrics", {}))
    prev_agg = cur.get("aggregate") if isinstance(cur, dict) else None
    if isinstance(prev_agg, (int, float)) and cand_agg is not None and cand_agg <= prev_agg:
        failed.append("does_not_beat_current_frontier")

    integ = _integrity(base_ref)
    if integ["status"] != "pass":
        failed.append("integrity")

    promotable = not failed
    return {
        "candidate": candidate,
        "promotable": promotable,
        "score": score["score"],
        "score_status": score["status"],
        "german_instruction_delta": d_gi,
        "candidate_aggregate": cand_agg,
        "previous_frontier_aggregate": prev_agg,
        "failed_gates": sorted(set(failed)),
        "score_detail": score,
        "integrity": integ["status"],
        "integrity_violations": integ["violations"],
        "candidate_summary": str(cand_summary),
        "baseline_summary": str(baseline_path),
        "model": run.get("model"),
    }


def write_frontier(verdict: Dict[str, Any]) -> None:
    """Write metadata pointers ONLY — no weights are copied or moved."""
    frontier = {
        "label": verdict["candidate"],
        "model": verdict["model"],
        "aggregate": verdict["candidate_aggregate"],
        "score": verdict["score"],
        "eval_summary": verdict["candidate_summary"],
        "baseline_summary": verdict["baseline_summary"],
        "commit": prov.current_git_commit(),
        "promoted_at": prov.now_iso(),
        "note": "metadata pointer only; weights live under outputs/posttrain/checkpoints/ (gitignored)",
    }
    FRONTIER.parent.mkdir(parents=True, exist_ok=True)
    FRONTIER.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def render(verdict: Dict[str, Any]) -> str:
    lines = ["# Post-training promotion report", "",
             f"Candidate: `{verdict['candidate']}`  ·  promotable: **{verdict['promotable']}**"]
    if verdict.get("error"):
        lines += ["", f"FAIL: {verdict['error']}"]
        return "\n".join(lines) + "\n"
    lines += ["",
              f"- score: `{verdict['score']}` ({verdict['score_status']})",
              f"- Δ german_instruction: `{verdict['german_instruction_delta']}` (must be > 0)",
              f"- candidate aggregate: `{verdict['candidate_aggregate']}`  ·  previous frontier: "
              f"`{verdict['previous_frontier_aggregate']}`",
              f"- integrity: `{verdict['integrity']}`"
              + (f" (violations: {verdict['integrity_violations']})"
                 if verdict["integrity_violations"] else ""),
              f"- failed gates: {verdict['failed_gates'] or 'none'}"]
    if verdict["promotable"]:
        lines += ["", "✅ Gates passed. frontier.json updated with metadata pointers (no weights). "
                  "Human review required before any public release."]
    else:
        lines += ["", "❌ Not promotable — frontier.json unchanged."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="latest")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--base-ref", default=None)
    ap.add_argument("--out", default=str(OUT / "promote"))
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    try:
        cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    except Exception:
        cfg = {}
    verdict = evaluate_promotion(args.candidate, cfg, pathlib.Path(args.baseline), args.base_ref)

    out_dir = pathlib.Path(args.out) / args.candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promotion_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "promotion_report.md").write_text(render(verdict), encoding="utf-8")

    if verdict["promotable"]:
        write_frontier(verdict)

    if args.format == "json":
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print(render(verdict))
    return 0 if verdict["promotable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
