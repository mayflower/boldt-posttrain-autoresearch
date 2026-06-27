#!/usr/bin/env python3
"""Train a DPO/ORPO/KTO-style preference specialist from German preference rows (dry-run-first).

Same auditable skeleton as ``pt_train_specialist.py`` but for preference optimization: the dry-run
plan surfaces the ``preference`` config block (loss_type, beta, length_ratio_max) so reviewers can
check the anti-response-suppression guard before a real run. Real mode fails closed until the
concrete preference trainer is implemented per the contracts.

    python scripts/pt_train_preference.py --config configs/posttrain/current.json \
        --specialist preference-de --out outputs/posttrain/runs --budget-minutes 90 --dry-run
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain import training  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--specialist", default="preference-de")
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/runs"))
    ap.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
    ap.add_argument("--budget-minutes", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--allow-checkpoints", action="store_true")
    args = ap.parse_args(argv)

    if args.real and args.dry_run:
        print("error: pass either --dry-run or --real, not both", file=sys.stderr)
        return 2
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    return training.run_training_trial(
        cfg=cfg, kind="preference", specialist=args.specialist, out_root=pathlib.Path(args.out),
        budget_minutes=args.budget_minutes, argv=argv, dry_run=dry, allow_gpu=args.allow_gpu,
        allow_checkpoints=args.allow_checkpoints, data_dir=pathlib.Path(args.data),
        config_errors=cfgmod.validate_config_dict(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
