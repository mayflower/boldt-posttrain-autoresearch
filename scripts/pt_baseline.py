#!/usr/bin/env python3
"""Establish (or inspect) the baseline German-core eval for the seed model (dry-run-first).

Writes the baseline summary FLAT at ``outputs/posttrain/baseline/summary.json`` (where pt_score /
pt_promote read it) plus a run card. Dry mode emits unmeasured plumbing; real mode requires
``--allow-gpu`` + the eval stack and fails closed until the German-core harness is implemented — a
dry baseline can never satisfy the scorer's "real measured baseline" gate.

    python scripts/pt_baseline.py --config configs/posttrain/current.json \
        --out outputs/posttrain/baseline --dry-run
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
from boldt_posttrain import provenance as prov  # noqa: E402
from boldt_posttrain import recipe  # noqa: E402

CONTRACT = "docs/posttrain-script-contracts.md"


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/baseline"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="baseline-seed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    args = ap.parse_args(argv)

    if args.real and args.dry_run:
        print("error: pass either --dry-run or --real, not both", file=sys.stderr)
        return 2
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    suite = cfg.get("eval", {}).get("suite", "german-core")
    model = args.model or cfg.get("training", {}).get("base_model")
    out_dir = pathlib.Path(args.out)
    git = recipe.persist_inputs(out_dir, cfg, argv)

    if dry:
        summary = recipe.eval_summary(model=model, label=args.label,
                                      metrics=recipe.metrics_skeleton(cfg), dry_run=True,
                                      suite=suite, note="baseline plumbing — not a measured baseline")
        status = "ok"
    elif not args.allow_gpu:
        summary = recipe.eval_summary(model=model, label=args.label,
                                      metrics=recipe.metrics_skeleton(cfg), dry_run=False,
                                      suite=suite, status="fail",
                                      note="--real requires --allow-gpu (human hardware gate)")
        status = "fail"
    else:
        stack_err = recipe.require_real_stack()
        note = (stack_err + "  (eval also needs: pip install -e '.[eval]')") if stack_err \
            else recipe.real_not_implemented("german_core_baseline", CONTRACT)["message"]
        summary = recipe.eval_summary(model=model, label=args.label,
                                      metrics=recipe.metrics_skeleton(cfg), dry_run=False,
                                      suite=suite, status="fail", note=note)
        status = "fail"

    summary["commit"] = git["commit"]
    recipe.write_json(out_dir / "summary.json", summary)
    card = prov.new_run_card(args.label, "baseline", "python " + " ".join(argv), model=model,
                             metrics=summary["metrics"],
                             output_artifacts=[str(out_dir / "summary.json")],
                             notes=summary.get("note", ""))
    prov.write_run_card(card, out_dir)

    print(json.dumps({"status": status, "mode": summary["mode"], "model": model,
                      "out": str(out_dir / "summary.json")}, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
