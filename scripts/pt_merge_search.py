#!/usr/bin/env python3
"""Merge search over complementary, compatible specialist checkpoints (dry-run-first).

Dry mode scans ``outputs/posttrain/runs/*/run_card.json`` for eligible training runs (same
warm-start basin), enumerates a merge MATRIX (candidate pairs × configured methods) with verdict
``needs_eval``, and writes ``merge/<merge_id>/merge_matrix.json``. Real mode requires the optional
``merge`` extra (mergekit) and fails closed until the concrete merge is implemented per the
contracts — it never claims a merged checkpoint it did not produce.

    python scripts/pt_merge_search.py --config configs/posttrain/current.json \
        --runs outputs/posttrain/runs --out outputs/posttrain/merge --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain import provenance as prov  # noqa: E402
from boldt_posttrain import recipe  # noqa: E402

CONTRACT = "docs/posttrain-script-contracts.md"


def _eligible(runs_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Training runs that are merge-eligible: a run card with a train_* run_type + a base model."""
    out: List[Dict[str, Any]] = []
    if not runs_dir.exists():
        return out
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        card_p = d / "run_card.json"
        if not card_p.exists():
            continue
        try:
            card = json.loads(card_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(card.get("run_type", "")).startswith("train_"):
            out.append({"run_id": card.get("run_id", d.name), "base_model": card.get("model"),
                        "run_type": card.get("run_type")})
    return out


def build_matrix(eligible: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    """Enumerate candidate merges: pairs sharing a base model × each configured method."""
    candidates: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(eligible, 2):
        if a["base_model"] != b["base_model"]:
            continue  # only merge descendants of the same warm-start basin
        for method in methods:
            candidates.append({
                "candidate": f"{a['run_id']}+{b['run_id']}::{method}",
                "parents": [a["run_id"], b["run_id"]],
                "method": method,
                "parameters": {},
                "eval_summary": None,
                "verdict": "needs_eval",
            })
    return candidates


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--runs", default=str(ROOT / "outputs/posttrain/runs"))
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/merge"))
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    args = ap.parse_args(argv)
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    methods = cfg.get("merge", {}).get("methods", ["linear", "slerp", "ties", "dare_ties"])
    eligible = _eligible(pathlib.Path(args.runs))
    candidates = build_matrix(eligible, methods)

    merge_id = f"merge-{'dry' if dry else 'real'}-{prov.stamp()}"
    out_dir = pathlib.Path(args.out) / merge_id

    matrix: Dict[str, Any] = {
        "merge_id": merge_id, "mode": "dry_run" if dry else "real",
        "base_model": cfg.get("merge", {}).get("base_model"),
        "methods": methods, "n_eligible": len(eligible),
        "eligible": eligible, "candidates": candidates,
        "note": "merge only same-basin descendants; verdict needs_eval until evaluated/scored.",
    }
    if dry:
        matrix["status"] = "ok"
        matrix["scale_disclaimer"] = "dry-run plumbing only — no merge was performed"
    else:
        try:
            import mergekit  # noqa: F401
            ni = recipe.real_not_implemented("merge_search", CONTRACT)
            matrix.update(status="fail", **{k: ni[k] for k in
                          ("missing_real_implementation", "message")})
        except Exception:
            matrix["status"] = "fail"
            matrix["message"] = "real merge needs the optional merge extra: pip install -e '.[merge]'"

    recipe.write_json(out_dir / "merge_matrix.json", matrix)
    if args.format == "markdown":
        print(f"# Merge search — {matrix['status']} ({matrix['mode']})\n")
        print(f"- eligible specialists: {len(eligible)}  ·  candidate merges: {len(candidates)}")
        print(f"- methods: {', '.join(methods)}  ·  out: `{out_dir / 'merge_matrix.json'}`")
        if len(eligible) < 2:
            print("\n_Fewer than 2 eligible specialists — train more branches before merging._")
        if matrix.get("message"):
            print(f"- {matrix['message']}")
    else:
        print(json.dumps({"status": matrix["status"], "merge_id": merge_id,
                          "n_eligible": len(eligible), "n_candidates": len(candidates),
                          "out": str(out_dir / "merge_matrix.json")}, ensure_ascii=False))
    return 0 if matrix["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
