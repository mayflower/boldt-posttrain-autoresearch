#!/usr/bin/env python3
"""Discover German candidate subsets or inspect explicitly policy-pinned sources (dry-run-first).

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
from boldt_posttrain import provenance as prov  # noqa: E402
from boldt_posttrain.data_pipeline import (  # noqa: E402
    canonical_json,
    discover_sources,
    language_identifier_from_config,
    sha256_bytes,
)

PLAN = [
    "inspect configured pinned sources, or list_datasets(author=data.org) when none are configured",
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
        doc = {
            "status": "ok" if not errors else "fail",
            "mode": "dry_run",
            "org": org,
            "candidates": [],
            "discovery_plan": PLAN,
            "config_errors": errors,
            "scale_disclaimer": "dry-run plumbing only — no datasets were inspected",
        }
    else:
        data_cfg = cfg.get("data", {})
        try:
            fixture = data_cfg.get("discovery_fixture")
            infos = (
                json.loads(pathlib.Path(fixture).read_text(encoding="utf-8"))
                if fixture
                else data_cfg.get("sources")
            )
            lang = language_identifier_from_config(
                data_cfg, cache_dir=ROOT / "outputs/posttrain/cache"
            )
            candidates = discover_sources(
                org=org,
                allowed_licenses=data_cfg.get("allowed_licenses", []),
                sample_size=int(data_cfg.get("discovery_sample_size", 64)),
                dataset_infos=infos,
                language_id=lang,
            )
            body = {
                "status": "ok",
                "mode": "real",
                "org": org,
                "run_id": f"discovery-{prov.stamp()}",
                "candidates": candidates,
                "language_id_hash": lang.model_hash,
            }
            doc = {**body, "artifact_hash": sha256_bytes(canonical_json(body))}
        except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            doc = {
                "status": "failed",
                "mode": "real",
                "org": org,
                "candidates": [],
                "message": f"technical discovery failure: {type(exc).__name__}: {exc}",
            }

    recipe.write_json(pathlib.Path(args.out), doc)
    if args.format == "markdown":
        print(f"# German data discovery — {doc['status']} ({doc['mode']})\n")
        print(f"- org: `{org}`  ·  candidates: {len(doc['candidates'])}  ·  out: `{args.out}`")
        for e in errors:
            print(f"- ✗ config: {e}")
        if doc.get("message"):
            print(f"- {doc['message']}")
        print("\n## Discovery plan\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(PLAN)))
    else:
        print(
            json.dumps(
                {
                    "status": doc["status"],
                    "org": org,
                    "candidates": len(doc["candidates"]),
                    "out": args.out,
                },
                ensure_ascii=False,
            )
        )
    return 0 if doc["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
