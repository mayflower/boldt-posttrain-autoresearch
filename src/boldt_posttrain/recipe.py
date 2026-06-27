"""Trial engine for the post-training loop (pure stdlib; ML imported lazily in --real paths only).

Centralizes what every lever script needs so they stay thin and consistent:

  * ``metrics_skeleton`` / ``eval_summary`` — the canonical metric SHAPE (matching the pt_eval
    contract) used for dry-run plumbing. Dry metrics are deliberately ``None`` (not ``0.0``) so
    the scorer's gates fail closed instead of mistaking plumbing for a measured result.
  * ``persist_inputs`` — write ``config.resolved.json`` / ``command.txt`` / ``env.json`` /
    ``git.status`` into a run dir, so every artifact is reproducible.
  * ``require_real_stack`` / ``real_not_implemented`` — the honest ``--real`` gate: real training,
    merging and evaluation need the optional ML stack AND a concrete implementation. Until that
    exists this fails closed with an actionable message rather than fabricating metrics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import provenance as prov

# The eval metric block contract (docs/posttrain-script-contracts.md :: pt_eval).
HELPFULNESS_KEYS = ["german_instruction", "format_following", "reasoning_core", "longcontext"]
RATE_KEYS = ["english_bleed_rate", "empty_output_rate", "refusal_rate", "over_refusal_rate"]


def deadline_after(budget_minutes: int) -> float:
    """Monotonic deadline ``budget_minutes`` from now (use ``time.monotonic()`` to compare)."""
    return time.monotonic() + max(0, budget_minutes) * 60


def metrics_skeleton(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Plumbing-only metric block: correct SHAPE, no measured values. Fails every score gate."""
    tasks = []
    if isinstance(cfg, dict):
        tasks = (cfg.get("eval", {}) or {}).get("lm_eval_tasks", []) or []
    m: Dict[str, Any] = {k: None for k in HELPFULNESS_KEYS}
    m.update({k: None for k in RATE_KEYS})
    m["safety"] = None
    m["lm_eval"] = {t: None for t in tasks}
    m["leakage"] = {"status": "not_checked", "hits": None}
    m["license"] = {"status": "unknown", "usable": False}
    return m


def eval_summary(*, model: Optional[str], label: str, metrics: Dict[str, Any], dry_run: bool,
                 suite: Optional[str] = None, artifacts: Optional[Dict[str, Any]] = None,
                 status: str = "ok", note: Optional[str] = None) -> Dict[str, Any]:
    """Build the canonical eval ``summary.json`` doc (matches the pt_eval output contract)."""
    doc: Dict[str, Any] = {
        "status": status,
        "model": model,
        "label": label,
        "mode": "dry_run" if dry_run else "real",
        "suite": suite,
        "metrics": metrics,
        "artifacts": dict(artifacts or {}),
    }
    if dry_run:
        doc["scale_disclaimer"] = ("dry-run plumbing only — metrics are unmeasured shape; "
                                   "never promotable")
    if note:
        doc["note"] = note
    return doc


def persist_inputs(out_dir: Path, cfg: Dict[str, Any], argv: List[str]) -> Dict[str, Any]:
    """Persist resolved config + command + env + git state into the run dir. Returns git info."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "command.txt").write_text(
        "python " + " ".join([sys.argv[0]] + list(argv)) + "\n", encoding="utf-8")
    (out_dir / "env.json").write_text(
        json.dumps(prov.collect_env_metadata(), ensure_ascii=False, indent=2), encoding="utf-8")
    git = prov.git_provenance()
    (out_dir / "git.status").write_text(git.get("status_short", ""), encoding="utf-8")
    return {"commit": git["commit"], "dirty": git["dirty"]}


def require_real_stack() -> Optional[str]:
    """Return None if the optional ML stack is importable, else an actionable error string."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return None
    except Exception:
        return ("--real needs the optional ML stack; install it and launch under the project env, "
                "e.g.:  pip install -e '.[train,eval]'  (then re-run with --real --allow-gpu)")


def real_not_implemented(feature: str, contract_ref: str) -> Dict[str, Any]:
    """Standardized fail doc for a real path that has no concrete implementation yet.

    Honest by construction: it claims no metrics. The loop treats this as ``needs-real`` and the
    operator implements ``feature`` per the named contract before a real run can produce numbers."""
    return {
        "status": "fail",
        "mode": "real",
        "missing_real_implementation": feature,
        "message": (f"real '{feature}' is not implemented in this scaffold. Implement it per "
                    f"{contract_ref}, then re-run. No metrics were produced (fail-closed)."),
    }


def write_json(path: Path, doc: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
