#!/usr/bin/env python3
"""Discover German candidate subsets in the Hugging Face org ``openeurollm`` (dry-run-first).

Dry mode validates config and writes a discovery skeleton + the documented discovery PLAN (how
German candidates are identified) without any network calls. Real mode requires the optional
``data`` extra (huggingface_hub/datasets) and fails closed until the concrete, offline-auditable
discovery is implemented per the contracts — it never invents dataset candidates.

    python scripts/pt_discover_openeurollm_de.py --config configs/posttrain/current.json \
        --out outputs/posttrain/data/discovery.json --format markdown --dry-run
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
PLAN = [
    "list_datasets(author='openeurollm') — no full downloads",
    "get_dataset_config_names / get_dataset_split_names per dataset",
    "flag German by config/split name (de|deu|ger|german|deutsch) OR a language column value "
    "in the allowlist OR a deterministic langid check on streamed samples",
    "guess schema (sft|preference|cpt) and record license (unknown => training_usable=false)",
]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/data/discovery.json"))
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args(argv)
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    errors = cfgmod.validate_config_dict(cfg)
    org = cfg.get("data", {}).get("org", "openeurollm")

    if dry:
        doc = {"status": "ok" if not errors else "fail", "mode": "dry_run", "org": org,
               "candidates": [], "discovery_plan": PLAN, "config_errors": errors,
               "scale_disclaimer": "dry-run plumbing only — no datasets were inspected"}
    else:
        try:
            import huggingface_hub  # noqa: F401
            ni = recipe.real_not_implemented("openeurollm_discovery", CONTRACT)
            doc = {"status": "fail", "mode": "real", "org": org, "candidates": [], **ni}
        except Exception:
            doc = {"status": "fail", "mode": "real", "org": org, "candidates": [],
                   "message": "real discovery needs the optional data extra: "
                              "pip install -e '.[data]'"}

    recipe.write_json(pathlib.Path(args.out), doc)
    if args.format == "markdown":
        print(f"# OpenEuroLLM German discovery — {doc['status']} ({doc['mode']})\n")
        print(f"- org: `{org}`  ·  candidates: {len(doc['candidates'])}  ·  out: `{args.out}`")
        for e in errors:
            print(f"- ✗ config: {e}")
        if doc.get("message"):
            print(f"- {doc['message']}")
        print("\n## Discovery plan\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(PLAN)))
    else:
        print(json.dumps({"status": doc["status"], "org": org,
                          "candidates": len(doc["candidates"]), "out": args.out},
                         ensure_ascii=False))
    return 0 if doc["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
