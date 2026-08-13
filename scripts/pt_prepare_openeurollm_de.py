#!/usr/bin/env python3
"""Materialize trainable German shards after license, language, and leakage checks.

Dry mode writes the manifest / leakage / quality REPORT skeletons (all fail-closed: not_trainable,
not_checked, unknown license) so the rest of the loop correctly refuses to train on them. Real mode
fails closed unless all verified discovery, selection, language-ID, and decontamination inputs
are present.

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
from boldt_posttrain.data_pipeline import (  # noqa: E402
    language_identifier_from_config,
    materialize_streaming,
    verify_selection_against_discovery,
    verify_hashed_artifact,
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--discovery", default=str(ROOT / "outputs/posttrain/data/discovery.json"))
    ap.add_argument("--selection", default=str(ROOT / "outputs/posttrain/data/selection.json"))
    ap.add_argument("--discovery-run-id", default=None)
    ap.add_argument("--selection-run-id", default=None)
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
        manifest = {
            "status": "not_trainable",
            "mode": "dry_run",
            "org": cfg.get("data", {}).get("org"),
            "discovery": args.discovery,
            "sources": [],
            "row_counts": {},
            "schemas": cfg.get("data", {}).get("schemas", []),
            "scale_disclaimer": "dry-run plumbing only — no shards materialized",
            "config_errors": errors,
        }
        leakage = {
            "status": "not_checked",
            "mode": "dry_run",
            "overlap_hits": None,
            "note": "fail-closed: training refuses until leakage is VERIFIED clean",
        }
        quality = {
            "status": "not_checked",
            "mode": "dry_run",
            "german_confidence": None,
            "dedup": None,
            "length_distribution": None,
            "refusal_safety_flags": None,
        }
        status = "ok" if not errors else "fail"
    else:
        try:
            if errors:
                raise ValueError("config invalid: " + "; ".join(errors))
            discovery = json.loads(pathlib.Path(args.discovery).read_text(encoding="utf-8"))
            selection = json.loads(pathlib.Path(args.selection).read_text(encoding="utf-8"))
            data_cfg = cfg.get("data", {})
            verify_selection_against_discovery(
                discovery,
                selection,
                allowed_org=str(data_cfg.get("org", "")),
                allowed_licenses=data_cfg.get("allowed_licenses", []),
                allowed_sources=data_cfg.get("allowed_sources", []),
            )
            if args.discovery_run_id != discovery.get("run_id"):
                raise ValueError("--discovery-run-id must exactly match discovery artifact")
            if args.selection_run_id != selection.get("run_id"):
                raise ValueError("--selection-run-id must exactly match selection artifact")
            language_id = language_identifier_from_config(
                data_cfg, cache_dir=ROOT / "outputs/posttrain/cache"
            )
            from boldt_posttrain.training import load_tokenizer

            training_cfg = cfg["training"]
            tokenizer = load_tokenizer(
                training_cfg["base_model"], revision=training_cfg.get("revision")
            )
            corpus_path = out / "decontamination.json"
            corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            if not verify_hashed_artifact(corpus):
                raise ValueError("decontamination corpus artifact hash is invalid")
            if corpus.get("policy_hash") != cfg.get("policy_hash"):
                raise ValueError("decontamination corpus policy hash is stale")
            manifest = materialize_streaming(
                selection.get("sources", []),
                out,
                language_id=language_id,
                min_german_confidence=float(data_cfg.get("min_german_confidence", 0.8)),
                exact_dedup_limit=int(data_cfg.get("exact_dedup_limit", 2_000_000)),
                decontamination_hash=corpus["artifact_hash"],
                decontamination_corpus=corpus,
                policy_hash=cfg.get("policy_hash"),
                discovery_run_id=args.discovery_run_id,
                selection_run_id=args.selection_run_id,
                seed=int(training_cfg["seed"]),
                max_rows_per_source=int(data_cfg.get("max_rows_per_source_real", 50_000)),
                global_max_rows=int(data_cfg.get("max_rows", 50_000)),
                validation_fraction=data_cfg["validation_fraction"],
                tokenizer=tokenizer,
                context_length=int(training_cfg["context_length"]),
                max_prompt_length=int(cfg["preference"]["max_prompt_length"]),
                max_completion_length=int(cfg["preference"]["max_completion_length"]),
            )
            leakage = json.loads((out / "leakage_report.json").read_text(encoding="utf-8"))
            quality = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
            status = "ok"
        except (OSError, ValueError, RuntimeError, MemoryError, json.JSONDecodeError) as exc:
            manifest = {
                "status": "not_trainable",
                "mode": "real",
                "sources": [],
                "message": f"technical preparation failure: {type(exc).__name__}: {exc}",
            }
            leakage = {"status": "not_checked", "mode": "real"}
            quality = {"status": "not_checked", "mode": "real"}
            status = "failed"

    published_artifacts: List[str] = []
    result_out = out
    if dry:
        # Dry-run skeletons must never replace a previously valid real publication.
        result_out = out / "dry-run"
        recipe.write_json(result_out / "manifest.json", manifest)
        recipe.write_json(result_out / "leakage_report.json", leakage)
        recipe.write_json(result_out / "quality_report.json", quality)
        published_artifacts = ["manifest.json", "leakage_report.json", "quality_report.json"]
    elif status == "ok":
        published_artifacts = ["manifest.json", "leakage_report.json", "quality_report.json"]

    result = {
        "status": status,
        "mode": manifest["mode"],
        "out": str(result_out),
        "artifacts": published_artifacts,
        "trainable": manifest.get("status") == "trainable",
    }
    if manifest.get("message"):
        result["message"] = manifest["message"]
    if args.format == "markdown":
        print(f"# German data prepare — {status} ({manifest['mode']})\n")
        print(
            f"- out: `{result_out}`  ·  trainable: **{result['trainable']}** (leakage {leakage['status']}, "
            f"manifest {manifest['status']})"
        )
        for e in errors:
            print(f"- ✗ config: {e}")
        if manifest.get("message"):
            print(f"- {manifest['message']}")
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if status == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
