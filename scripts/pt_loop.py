#!/usr/bin/env python3
"""Run ONE deterministic, non-agent post-training iteration — a scriptable CI gate.

Pipeline (reusing the canonical scripts, NO gate logic duplicated): eval -> score -> log ->
integrity. Prints a single machine-readable JSON verdict so an operator/CI can read the outcome
without an agent in the loop. The OUTER loop (edit ``configs/posttrain/current.json``, train a
specialist, re-run) is the agent's job under ``/pt-run``; this command is one auditable step.

Exit code is 0 ONLY when the eval ran, the score gate passed against a real baseline, AND the
integrity check passed — so it is safe to wire directly as a gate. Dry runs and missing baselines
can never be promotable (fail-closed), mirroring the embed loop's ``ar_loop.py``.

    # dry-run (stdlib only): plumbing check
    python scripts/pt_loop.py --candidate baseline-seed --dry-run

    # real iteration (eval the configured/!given model, then judge it)
    python scripts/pt_loop.py --real --allow-gpu --model outputs/posttrain/checkpoints/<run> \
        --label cand-1 --baseline outputs/posttrain/baseline/summary.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import provenance as prov  # noqa: E402

EVALS = ROOT / "outputs" / "posttrain" / "evals"
DEFAULT_BASELINE = ROOT / "outputs" / "posttrain" / "baseline" / "summary.json"
DEFAULT_RESULTS = ROOT / "outputs" / "posttrain" / "results.tsv"


def _load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "scripts" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quiet(fn, argv: List[str]) -> int:
    """Call a sub-script main(argv), swallowing its stdout so only our verdict is emitted."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(argv)


def _dig(d: Any, *path: str) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/posttrain/current.json"))
    ap.add_argument("--model", default=None, help="model id / checkpoint to evaluate (the trial)")
    ap.add_argument("--candidate", default=None, help="candidate label passed through to pt_eval")
    ap.add_argument("--label", default=None, help="output label under the evals dir")
    ap.add_argument("--evals-out", default=str(EVALS))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--budget-minutes", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--base-ref", default=None,
                    help="loop start commit; passed to integrity so committed protected edits "
                         "are caught")
    ap.add_argument("--status", default="keep", help="results.tsv disposition for this row")
    ap.add_argument("--notes", default="pt_loop iteration")
    args = ap.parse_args(argv)

    if args.real and args.dry_run:
        print("error: pass either --dry-run or --real, not both", file=sys.stderr)
        return 2
    dry = not args.real
    mode_flag = "--dry-run" if dry else "--real"
    label = args.label or args.candidate or f"loop-{prov.stamp()}"
    evals_out = pathlib.Path(args.evals_out)
    out_dir = evals_out / label
    summary_path = out_dir / "summary.json"

    pt_eval = _load("pt_eval")
    pt_score = _load("pt_score")
    pt_log = _load("pt_log_result")
    integ = _load("check_posttrain_integrity")

    # 1) trial = evaluate the candidate/model -------------------------------------------------
    eval_argv = ["--config", args.config, "--out", str(evals_out), "--label", label, mode_flag]
    if args.model:
        eval_argv += ["--model", args.model]
    if args.candidate:
        eval_argv += ["--candidate", args.candidate]
    if args.allow_gpu:
        eval_argv.append("--allow-gpu")
    eval_rc = _quiet(pt_eval.main, eval_argv)

    if not summary_path.exists():
        print(json.dumps({"label": label, "stage": "eval", "eval_rc": eval_rc,
                          "error": f"no summary.json at {summary_path} — eval failed before "
                                   "writing output", "promotable": False}, ensure_ascii=False,
                         indent=2))
        return eval_rc or 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # 2) score (only against a real baseline file) --------------------------------------------
    score_doc: Optional[Dict[str, Any]] = None
    baseline = pathlib.Path(args.baseline)
    if baseline.exists():
        _quiet(pt_score.main, ["--config", args.config, "--run", str(summary_path),
                               "--baseline", str(baseline), "--out", str(out_dir / "score.json"),
                               "--format", "json"])
        sp = out_dir / "score.json"
        if sp.exists():
            score_doc = json.loads(sp.read_text(encoding="utf-8"))

    # 3) log one auditable row ----------------------------------------------------------------
    _quiet(pt_log.main, ["--run", str(out_dir), "--results", args.results,
                         "--status", args.status, "--notes", args.notes])

    # 4) integrity (read the structured result directly) --------------------------------------
    integ_result = integ.evaluate(integ.changed_paths(args.base_ref))

    promotable = (summary.get("status") in ("ok", "pass")
                  and not summary.get("scale_disclaimer")
                  and summary.get("mode") == "real"
                  and integ_result["status"] == "pass"
                  and score_doc is not None and score_doc.get("status") == "pass")

    verdict = {
        "label": label,
        "out": str(out_dir),
        "mode": summary.get("mode"),
        "eval_status": summary.get("status"),
        "score_status": (score_doc or {}).get("status"),
        "score": (score_doc or {}).get("score"),
        "failed_gates": [g["name"] for g in (score_doc or {}).get("failed_gates", [])],
        "leakage_status": _dig(summary, "metrics", "leakage", "status"),
        "license_status": _dig(summary, "metrics", "license", "status"),
        "integrity": integ_result["status"],
        "integrity_violations": integ_result["violations"],
        "results_tsv": args.results,
        "baseline_present": baseline.exists(),
        "budget_minutes": args.budget_minutes,
        "promotable": promotable,
    }
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
