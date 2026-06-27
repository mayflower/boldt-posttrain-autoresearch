#!/usr/bin/env python3
"""Materialize trainable German shards — only after license/language/leakage checks (dry-run-first).

Dry mode writes the manifest / leakage / quality REPORT skeletons (all fail-closed: not_trainable,
not_checked, unknown license) so the rest of the loop correctly refuses to train on them. Real mode
fails closed until the concrete preparer is implemented per the contracts — it never writes a
"clean" leakage report or a usable license it did not actually verify.

    python scripts/pt_prepare_openeurollm_de.py --config configs/posttrain/current.json \
        --discovery outputs/posttrain/data/discovery.json --out outputs/posttrain/data --dry-run
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
from boldt_posttrain import recipe  # noqa: E402

CONTRACT = "docs/posttrain-script-contracts.md"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--discovery", default=str(ROOT / "outputs/posttrain/data/discovery.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/data"))
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args(argv)
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    errors = cfgmod.validate_config_dict(cfg)
    out = pathlib.Path(args.out)

    if dry:
        manifest = {"status": "not_trainable", "mode": "dry_run", "org": cfg.get("data", {}).get("org"),
                    "discovery": args.discovery, "sources": [], "row_counts": {},
                    "schemas": cfg.get("data", {}).get("schemas", []),
                    "scale_disclaimer": "dry-run plumbing only — no shards materialized",
                    "config_errors": errors}
        leakage = {"status": "not_checked", "mode": "dry_run", "overlap_hits": None,
                   "note": "fail-closed: training refuses until leakage is VERIFIED clean"}
        quality = {"status": "not_checked", "mode": "dry_run", "german_confidence": None,
                   "dedup": None, "length_distribution": None, "refusal_safety_flags": None}
        status = "ok" if not errors else "fail"
    else:
        ni = recipe.real_not_implemented("openeurollm_prepare", CONTRACT)
        manifest = {"status": "not_trainable", "mode": "real", "sources": [], **ni}
        leakage = {"status": "not_checked", "mode": "real",
                   "note": "fail-closed: real preparer not implemented; no clean status claimed"}
        quality = {"status": "not_checked", "mode": "real"}
        status = "fail"

    recipe.write_json(out / "manifest.json", manifest)
    recipe.write_json(out / "leakage_report.json", leakage)
    recipe.write_json(out / "quality_report.json", quality)

    result = {"status": status, "mode": manifest["mode"], "out": str(out),
              "artifacts": ["manifest.json", "leakage_report.json", "quality_report.json"],
              "trainable": False}
    if args.format == "markdown":
        print(f"# OpenEuroLLM German prepare — {status} ({manifest['mode']})\n")
        print(f"- out: `{out}`  ·  trainable: **False** (leakage {leakage['status']}, "
              f"manifest {manifest['status']})")
        for e in errors:
            print(f"- ✗ config: {e}")
        if manifest.get("message"):
            print(f"- {manifest['message']}")
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
