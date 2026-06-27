"""Config resolution for the post-training loop (pure stdlib).

A config may declare ``"extends": "<path-relative-to-repo-root>"``; ``resolve_config`` deep-merges
the base under the overlay so ``configs/posttrain/current.json`` inherits ``base.json`` defaults.
The merged dict records ``_extends`` for provenance. ``validate_config_dict`` performs cheap,
fail-closed structural checks used by ``pt_status`` and the dry-run paths — it never imports ML.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "posttrain" / "current.json"
BASE_CONFIG = ROOT / "configs" / "posttrain" / "base.json"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; nested dicts merged)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load a config, merging the base referenced via ``extends`` (path relative to repo root)."""
    config_path = pathlib.Path(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    extends = cfg.get("extends")
    if isinstance(extends, str):
        base = json.loads((ROOT / extends).read_text(encoding="utf-8"))
        merged = deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
        merged["_extends"] = extends
        return merged
    return cfg


def validate_config_dict(cfg: Dict[str, Any]) -> List[str]:
    """Cheap structural validation of a RESOLVED config. Returns a list of human-readable errors
    (empty == valid). Fail-closed: missing blocks are errors, not silently defaulted."""
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["config must be a JSON object"]

    training = cfg.get("training")
    if not isinstance(training, dict):
        errors.append("missing 'training' block")
    else:
        if not training.get("base_model"):
            errors.append("training.base_model is required (the protected Boldt seed)")
        method = training.get("method")
        if method not in ("qlora", "lora", "full", None):
            errors.append(f"training.method '{method}' not in qlora|lora|full")

    data = cfg.get("data")
    if not isinstance(data, dict):
        errors.append("missing 'data' block")
    else:
        if data.get("org") != "openeurollm":
            errors.append("data.org must be 'openeurollm' (the only default remote source family)")
        if not data.get("language_allowlist"):
            errors.append("data.language_allowlist is required for German filtering")

    ev = cfg.get("eval")
    if not isinstance(ev, dict):
        errors.append("missing 'eval' block")
    elif not ev.get("lm_eval_tasks"):
        errors.append("eval.lm_eval_tasks is required for German-core regression checks")

    return errors


def load_resolved(config_path: Optional[str] = None) -> Dict[str, Any]:
    return resolve_config(pathlib.Path(config_path) if config_path else DEFAULT_CONFIG)
