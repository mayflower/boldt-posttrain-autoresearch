#!/usr/bin/env python3
"""Evaluate a seed / specialist / merged candidate on the German-core suite (dry-run-first).

Writes ``outputs/posttrain/evals/<label>/summary.json`` in the canonical metric shape (the pt_eval
contract). Dry mode emits unmeasured plumbing (never promotable). Real mode requires ``--allow-gpu``
and the optional ``eval`` extra (lm-eval), then fails closed until the concrete German-core harness
is implemented per the contracts — it never fabricates scores.

    python scripts/pt_eval.py --config configs/posttrain/current.json --candidate latest \
        --out outputs/posttrain/evals --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain import provenance as prov  # noqa: E402
from boldt_posttrain import recipe  # noqa: E402
from boldt_posttrain.evaluation import (  # noqa: E402
    attach_paired_intervals,
    classify_exception,
    finalize_summary,
    resolve_suite,
    run_real_evaluation,
)


def _label(args) -> str:
    if args.label:
        return args.label
    if args.candidate:
        return args.candidate
    if args.model:
        return prov.slug(pathlib.Path(args.model).name)
    return "eval"


def run_eval(
    *,
    cfg,
    model,
    candidate,
    label,
    out_dir,
    argv,
    dry_run,
    allow_gpu,
    profile="dev",
    device="cuda:0",
    suite_path=None,
    budget_minutes=90,
):
    suite = cfg.get("eval", {}).get("suite", "german-core")
    out_dir = pathlib.Path(out_dir)
    git = recipe.persist_inputs(out_dir, cfg, argv)
    command = "python " + " ".join(argv)

    if dry_run:
        metrics = recipe.metrics_skeleton(cfg)
        summary = recipe.eval_summary(
            model=model,
            label=label,
            metrics=metrics,
            dry_run=True,
            suite=suite,
            note=f"candidate={candidate}",
        )
        summary["profile"] = profile
        summary["technical_error_count"] = 0
        summary["model_error_count"] = 0
        status = "ok"
    elif device.startswith("cuda:") and not allow_gpu:
        summary = recipe.eval_summary(
            model=model,
            label=label,
            metrics=recipe.metrics_skeleton(cfg),
            dry_run=False,
            suite=suite,
            status="fail",
            note="--real requires --allow-gpu (human hardware gate)",
        )
        status = "fail"
    else:
        try:
            summary = run_real_evaluation(
                model_ref=model,
                suite_path=pathlib.Path(suite_path),
                profile=profile,
                device=device,
                output_dir=out_dir,
                config=cfg,
                deadline=time.monotonic() + budget_minutes * 60,
            )
            if profile == "promotion" and summary.get("status") == "ok":
                baseline_path = ROOT / "outputs/posttrain/baseline/promotion/current.json"
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                candidate_rows = json.loads(
                    pathlib.Path(summary["artifacts"]["raw_generations"]).read_text(
                        encoding="utf-8"
                    )
                )
                baseline_rows = json.loads(
                    pathlib.Path(baseline["artifacts"]["raw_generations"]).read_text(
                        encoding="utf-8"
                    )
                )
                attach_paired_intervals(summary, candidate_rows, baseline_rows)
            status = summary["status"]
        except (RuntimeError, ValueError, OSError) as exc:
            error_kind = classify_exception(exc)
            summary = recipe.eval_summary(
                model=model,
                label=label,
                metrics=recipe.metrics_skeleton(cfg),
                dry_run=False,
                suite=suite,
                status="failed",
                note=f"technical evaluation failure: {type(exc).__name__}: {exc}",
            )
            summary.update(
                profile=profile,
                technical_error_count=1,
                model_error_count=0,
                technical_errors={error_kind: 1},
            )
            status = "failed"

    summary["commit"] = git["commit"]
    summary["config_path"] = suite
    finalize_summary(summary)
    recipe.write_json(out_dir / "summary.json", summary)
    card = prov.new_run_card(
        label,
        "eval",
        command,
        model=model,
        metrics=summary["metrics"],
        output_artifacts=[str(out_dir / "summary.json")],
        notes=summary.get("note", ""),
    )
    prov.write_run_card(card, out_dir)
    return status, out_dir, summary


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--model", default=None, help="model id / checkpoint path to evaluate")
    ap.add_argument("--candidate", default=None, help="candidate label (e.g. latest / a run id)")
    ap.add_argument("--label", default=None, help="output label under the evals dir")
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/evals"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--profile", choices=["proxy", "dev", "promotion"], default="dev")
    ap.add_argument("--suite", default=None, help="explicit local suite path for dev/proxy tests")
    ap.add_argument("--budget-minutes", type=int, default=90)
    args = ap.parse_args(argv)

    if args.real and args.dry_run:
        print("error: pass either --dry-run or --real, not both", file=sys.stderr)
        return 2
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    label = _label(args)
    model = args.model or cfg.get("training", {}).get("base_model")
    if args.suite:
        suite_path = pathlib.Path(args.suite)
    elif dry:
        suite_path = ROOT / cfg.get("eval", {}).get("dev_suite", "data/eval/dev.json")
    else:
        suite_path = resolve_suite(
            args.profile,
            dev_path=ROOT / cfg.get("eval", {}).get("dev_suite", "data/eval/dev.json"),
            promotion_registry=ROOT
            / cfg.get("eval", {}).get(
                "promotion_registry", "configs/posttrain/promotion-suite.json"
            ),
        )
    status, out_dir, summary = run_eval(
        cfg=cfg,
        model=model,
        candidate=args.candidate,
        label=label,
        out_dir=pathlib.Path(args.out) / label,
        argv=argv,
        dry_run=dry,
        allow_gpu=args.allow_gpu,
        profile=args.profile,
        device=args.device,
        suite_path=suite_path,
        budget_minutes=args.budget_minutes,
    )

    print(
        json.dumps(
            {
                "status": status,
                "mode": summary["mode"],
                "label": label,
                "out": str(out_dir / "summary.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if status == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
