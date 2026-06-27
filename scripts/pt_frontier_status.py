#!/usr/bin/env python3
"""Frontier-program state the autonomous loop reads each round (pure stdlib, read-only).

Scans saved ``outputs/posttrain/evals/<label>/summary.json`` and reports the German-helpfulness
aggregate per candidate, the current frontier-best (real runs only), and the per-dimension leaders
— i.e. which complementary specialists are worth MERGING. Never trains, never claims beyond saved
summaries. ``scripts/pt_promote.py`` is the actual gate that writes ``frontier.json``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import frontier as fr  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", default=str(fr.EVALS))
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    view = fr.build_frontier(pathlib.Path(args.evals))
    if args.format == "json":
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return 0

    def f(x):
        return "  .  " if not isinstance(x, (int, float)) else f"{x:.4f}"

    print("# Post-training frontier — German-helpfulness aggregate (saved eval summaries)\n")
    if view["n_candidates"] == 0:
        print("No eval summaries yet under outputs/posttrain/evals/. "
              "Run /pt-baseline then /pt-eval.")
        return 0
    print("| candidate | mode | agg | " + " | ".join(fr.DIMS) + " |")
    print("|---|---|---:|" + "---:|" * len(fr.DIMS))
    for c in view["candidates"]:
        print(f"| {c['label']} | {c['mode']} | {f(c['aggregate'])} | "
              + " | ".join(f(c["dims"].get(d)) for d in fr.DIMS) + " |")
    best = view["frontier_best"]
    if best:
        print(f"\nFrontier-best (real): `{best['label']}` agg {f(best['aggregate'])}")
    else:
        print("\nNo REAL evaluated candidate yet — frontier-best is undefined "
              "(dry-run candidates never rank).")
    if len(view["complementary_merge_inputs"]) >= 2:
        print("\n→ Complementary checkpoints to MERGE: "
              + ", ".join("`" + d + "`" for d in view["complementary_merge_inputs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
