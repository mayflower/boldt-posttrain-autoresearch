"""Provenance helpers: env capture, git state, and run cards (pure stdlib).

A run card is a small JSON record so any number under ``outputs/posttrain/`` traces back to the
exact command, commit, environment, inputs, and outputs that produced it. Package versions are read
from *metadata* (``importlib.metadata``), so collecting env info imports no ML. This mirrors the
inspiration repo's ``experiment_registry`` but targets the post-training (LoRA/merge/eval) stack.
"""
from __future__ import annotations

import importlib.metadata as ilm
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

RUN_TYPES = {"data", "baseline", "train_specialist", "train_preference", "train_cpt",
             "merge", "eval", "promote"}
REQUIRED_FIELDS = ("run_id", "run_type", "command", "commit", "date")

# Packages that matter for post-training provenance.
_TRACKED_PKGS = ("torch", "transformers", "trl", "peft", "accelerate", "datasets",
                 "bitsandbytes", "mergekit", "lm-eval")


def current_git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def git(cmd: List[str]) -> str:
    """Run a git subcommand, returning stdout ("" on any failure). Bounded timeout."""
    try:
        out = subprocess.run(["git"] + cmd, cwd=str(ROOT), capture_output=True,
                             text=True, timeout=15)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def pkg_version(pkg: str) -> Optional[str]:
    try:
        return ilm.version(pkg)  # reads metadata; does NOT import the package
    except Exception:
        return None


def collect_env_metadata() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "commit": current_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for pkg in _TRACKED_PKGS:
        env[pkg.replace("-", "_")] = pkg_version(pkg)
    return env


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def git_provenance() -> Dict[str, Any]:
    """Commit + dirty flag for embedding into metrics/run docs (best-effort)."""
    commit = current_git_commit()
    commit = None if commit in ("", "unknown") else commit
    status_short = git(["status", "--short"])
    return {"commit": commit, "dirty": bool(status_short.strip()), "status_short": status_short}


def new_run_card(run_id: str, run_type: str, command: str, *, model: Optional[str] = None,
                 dataset: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None, data_manifest: Optional[str] = None,
                 input_artifacts: Optional[Sequence[str]] = None,
                 output_artifacts: Optional[Sequence[str]] = None, notes: str = "",
                 env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env = env or collect_env_metadata()
    card = {
        "run_id": run_id,
        "run_type": run_type,
        "command": command,
        "commit": env.get("commit", "unknown"),
        "date": now_iso(),
        "hardware": env.get("platform"),
        "seed": seed,
        "model": model,
        "dataset": dataset,
        "data_manifest": data_manifest,
        "input_artifacts": list(input_artifacts or []),
        "output_artifacts": list(output_artifacts or []),
        "metrics": dict(metrics or {}),
        "env": {k: env.get(k) for k in
                ("python", "torch", "transformers", "trl", "peft", "datasets", "mergekit")},
        "notes": notes,
    }
    return card


def validate_run_card(card: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(card, dict):
        return ["run card must be a JSON object"]
    for f in REQUIRED_FIELDS:
        if not card.get(f):
            errors.append(f"missing required field '{f}'")
    if card.get("run_type") not in RUN_TYPES:
        errors.append(f"run_type '{card.get('run_type')}' not in {sorted(RUN_TYPES)}")
    if "metrics" in card and not isinstance(card["metrics"], dict):
        errors.append("'metrics' must be an object")
    return errors


def write_run_card(card: Dict[str, Any], out_dir: Path) -> str:
    errors = validate_run_card(card)
    if errors:
        raise ValueError("invalid run card: " + "; ".join(errors))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_card.json"
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
