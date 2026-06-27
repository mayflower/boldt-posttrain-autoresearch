#!/usr/bin/env python3
"""Combined loop report: readiness + frontier + recent results (pure stdlib, read-only by default).

Aggregates ``pt_status`` (readiness/next-lever), ``frontier`` (the candidate ranking), and the tail
of ``results.tsv`` (the audit log) into one view. ``--no-write`` prints only; otherwise it also
writes ``outputs/posttrain/report.md`` (+ ``report.json``) for the record.
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

import importlib.util  # noqa: E402


def _load_status():
    spec = importlib.util.spec_from_file_location("pt_status", ROOT / "scripts" / "pt_status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tail_results(n: int) -> List[str]:
    p = ROOT / "outputs" / "posttrain" / "results.tsv"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return lines
    return [lines[0]] + lines[1:][-n:]  # header + last n DATA rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "posttrain" / "current.json"))
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--tail", type=int, default=10)
    args = ap.parse_args(argv)

    status = _load_status().assess(args.config)
    view = fr.build_frontier()
    current = fr.current_frontier()
    results_tail = _tail_results(args.tail)

    doc = {"status": status, "frontier": view, "promoted_frontier": current,
           "results_tail": results_tail}

    if args.format == "json":
        out = json.dumps(doc, ensure_ascii=False, indent=2)
    else:
        lines = ["# Post-training AutoResearch — report", "",
                 f"Base model: `{status['base_model']}`  ·  "
                 f"config {'valid' if status['config_valid'] else 'INVALID'}  ·  "
                 f"scripts {status['scripts_present']}/{status['scripts_total']}", "",
                 "## Artifacts", ""]
        lines += [f"- {'✓' if v else '·'} {k}" for k, v in status["artifacts"].items()]
        lines += ["", "## Frontier", ""]
        if view["n_candidates"] == 0:
            lines.append("_No eval summaries yet._")
        else:
            best = view["frontier_best"]
            lines.append(f"- candidates: {view['n_candidates']} (real: {view['n_real']})")
            lines.append(f"- frontier-best (real): "
                         + (f"`{best['label']}` agg {best['aggregate']}" if best else "none"))
            if len(view["complementary_merge_inputs"]) >= 2:
                lines.append("- complementary merge inputs: "
                             + ", ".join("`" + d + "`" for d in view["complementary_merge_inputs"]))
        if current:
            lines += ["", "## Promoted frontier (frontier.json)", "",
                      f"```json\n{json.dumps(current, ensure_ascii=False, indent=2)}\n```"]
        lines += ["", "## Recent results", ""]
        if results_tail:
            lines += ["```", *results_tail, "```"]
        else:
            lines.append("_No results.tsv rows yet._")
        lines += ["", f"**Next lever:** {status['next_lever']}"]
        out = "\n".join(lines) + "\n"

    print(out)
    if not args.no_write:
        outdir = ROOT / "outputs" / "posttrain"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "report.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        if args.format == "markdown":
            (outdir / "report.md").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
