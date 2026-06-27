#!/usr/bin/env python3
"""Read-only readiness + artifact summary for the post-training loop (pure stdlib).

Reports config validity, which canonical scripts exist, which on-disk artifacts are present, and the
single highest-value NEXT lever — the view ``/pt-orient`` and ``/pt-status`` surface. Never edits.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402

OUT = ROOT / "outputs" / "posttrain"

CANONICAL_SCRIPTS = [
    "pt_status", "pt_report", "pt_frontier_status", "check_posttrain_integrity",
    "pt_discover_openeurollm_de", "pt_prepare_openeurollm_de", "pt_baseline",
    "pt_train_specialist", "pt_train_preference", "pt_train_cpt", "pt_merge_search",
    "pt_eval", "pt_score", "pt_promote", "pt_log_result",
]


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _summary_is_real(rel: str) -> bool:
    """True only for a REAL measured eval summary (dry plumbing does not count)."""
    p = ROOT / rel
    if not p.exists():
        return False
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return doc.get("mode") == "real" and not doc.get("scale_disclaimer") \
        and doc.get("status") in ("ok", "pass")


def assess(config_path: str) -> Dict[str, Any]:
    scripts = {name: _exists(f"scripts/{name}.py") for name in CANONICAL_SCRIPTS}
    artifacts = {
        "data_manifest": _exists("outputs/posttrain/data/manifest.json"),
        "data_discovery": _exists("outputs/posttrain/data/discovery.json"),
        "leakage_report": _exists("outputs/posttrain/data/leakage_report.json"),
        "baseline_summary": _exists("outputs/posttrain/baseline/summary.json"),
        "frontier": _exists("outputs/posttrain/frontier.json"),
        "results_tsv": _exists("outputs/posttrain/results.tsv"),
    }
    runs = sorted(p.name for p in (OUT / "runs").glob("*")) if (OUT / "runs").exists() else []
    evals = sorted(p.name for p in (OUT / "evals").glob("*")
                   if (p / "summary.json").exists()) if (OUT / "evals").exists() else []
    baseline_real = _summary_is_real("outputs/posttrain/baseline/summary.json")
    evals_real = [lbl for lbl in evals
                  if _summary_is_real(f"outputs/posttrain/evals/{lbl}/summary.json")]

    cfg_errors: List[str] = []
    try:
        cfg = cfgmod.resolve_config(pathlib.Path(config_path))
        cfg_errors = cfgmod.validate_config_dict(cfg)
        base_model = cfg.get("training", {}).get("base_model")
    except Exception as exc:
        cfg_errors = [f"could not resolve config: {exc}"]
        base_model = None

    # next lever (deterministic, matches AUTORESEARCH_POSTTRAIN.md priority)
    if cfg_errors:
        nxt = "fix configs/posttrain/current.json (see config_errors), then /pt-status"
    elif not all(scripts.values()):
        nxt = "/pt-bootstrap  (missing scripts: " \
              + ", ".join(n for n, ok in scripts.items() if not ok) + ")"
    elif not artifacts["data_manifest"]:
        nxt = "/pt-data dry   (no clean German OpenEuroLLM manifest yet)"
    elif not baseline_real:
        nxt = "/pt-baseline real --allow-gpu   (no REAL measured baseline yet)"
    elif not evals_real:
        nxt = "/pt-train real <specialist>  then  /pt-eval real latest"
    else:
        nxt = "/pt-merge real  or  /pt-promote <best-candidate>"

    return {
        "config_path": config_path,
        "base_model": base_model,
        "config_valid": not cfg_errors,
        "config_errors": cfg_errors,
        "scripts_present": sum(scripts.values()),
        "scripts_total": len(scripts),
        "missing_scripts": [n for n, ok in scripts.items() if not ok],
        "artifacts": artifacts,
        "baseline_real": baseline_real,
        "runs": runs,
        "evals": evals,
        "evals_real": evals_real,
        "next_lever": nxt,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    s = assess(args.config)
    if args.format == "json":
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return 0

    print("# Post-training AutoResearch — status\n")
    print(f"- base model: `{s['base_model']}`")
    print(f"- config: `{s['config_path']}` — {'valid' if s['config_valid'] else 'INVALID'}")
    for e in s["config_errors"]:
        print(f"  - ✗ {e}")
    print(f"- scripts implemented: {s['scripts_present']}/{s['scripts_total']}"
          + (f" (missing: {', '.join(s['missing_scripts'])})" if s["missing_scripts"] else ""))
    print("- artifacts:")
    for k, v in s["artifacts"].items():
        suffix = ""
        if k == "baseline_summary" and v:
            suffix = "  (REAL)" if s["baseline_real"] else "  (dry plumbing — not a measured baseline)"
        print(f"  - {'✓' if v else '·'} {k}{suffix}")
    print(f"- runs: {len(s['runs'])}  ·  evaluated candidates: {len(s['evals'])} "
          f"({len(s['evals_real'])} real)"
          + (f" — {', '.join(s['evals'])}" if s["evals"] else ""))
    print(f"\n**Next lever:** {s['next_lever']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
