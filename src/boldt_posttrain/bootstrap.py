"""Deterministic bootstrap derived solely from verified artifacts."""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .data_pipeline import IntegrityError, make_selection_artifact, verify_hashed_artifact


class BootstrapState(IntEnum):
    EMPTY = 0
    DISCOVERED = 1
    DATA_READY = 2
    BASELINE_READY = 3
    RESEARCH_READY = 4


def _read(path: Path) -> Optional[Dict[str, Any]]:
    if not Path(path).is_file():
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verified(document: Optional[Mapping[str, Any]], statuses: Sequence[str]) -> bool:
    return bool(
        document
        and document.get("status") in statuses
        and document.get("mode") == "real"
        and verify_hashed_artifact(document)
    )


def derive_state(output_root: Path) -> BootstrapState:
    root = Path(output_root)
    discovery = _read(root / "data" / "discovery.json")
    if discovery is not None and not _verified(discovery, ("ok", "verified")):
        raise IntegrityError("existing discovery artifact failed verification")
    if not _verified(discovery, ("ok", "verified")):
        return BootstrapState.EMPTY
    selection = _read(root / "data" / "selection.json")
    manifest = _read(root / "data" / "manifest.json")
    leakage = _read(root / "data" / "leakage_report.json")
    if selection is not None and not _verified(selection, ("ok", "verified")):
        raise IntegrityError("existing selection artifact failed verification")
    if manifest is not None and not _verified(manifest, ("trainable", "ok")):
        raise IntegrityError("existing data manifest failed verification")
    if not (
        _verified(selection, ("ok", "verified"))
        and _verified(manifest, ("trainable", "ok"))
        and leakage
        and leakage.get("status") in {"clean", "verified_clean"}
    ):
        return BootstrapState.DISCOVERED
    baseline = _read(root / "baseline" / "dev" / "current.json")
    if baseline is None:
        # W01 compatibility with the original baseline location.
        baseline = _read(root / "baseline" / "summary.json")
    if baseline is not None and not verify_hashed_artifact(baseline):
        raise IntegrityError("existing development baseline failed verification")
    if not (
        baseline
        and baseline.get("mode") == "real"
        and baseline.get("status") in {"ok", "pass"}
        and int(baseline.get("technical_error_count", 0)) == 0
    ):
        return BootstrapState.DATA_READY
    # BASELINE_READY is distinct from RESEARCH_READY: the configured default lever must have
    # actual compatible input material, not merely a baseline pointer.
    try:
        from .config import load_resolved

        lever = str(load_resolved().get("experiment", {}).get("lever", "sft"))
    except (OSError, ValueError, json.JSONDecodeError):
        return BootstrapState.BASELINE_READY
    schemas = {source.get("schema") for source in selection.get("sources", [])}
    schemas.update(shard.get("schema") for shard in manifest.get("shards", []))
    required_schema = {
        "sft": "sft",
        "preference": "preference",
        "cpt": "cpt",
        "rlvr": "rlvr",
        "grpo": "verified_math",
    }
    if lever in required_schema and required_schema[lever] not in schemas:
        return BootstrapState.BASELINE_READY
    if lever not in {*required_schema, "merge"}:
        return BootstrapState.BASELINE_READY
    return BootstrapState.RESEARCH_READY


def run_bootstrap(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    discover: Callable[[], Mapping[str, Any]],
    prepare: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    baseline: Callable[[], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Advance an empty checkout to research readiness without a separate state file.

    The callables are the existing discovery, preparation, and baseline domain operations.  Each
    returned artifact is required to be real and hashed before the next operation begins.
    """
    root = Path(output_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    state = derive_state(root)
    actions = []
    if state == BootstrapState.EMPTY:
        discovery = dict(discover())
        if not _verified(discovery, ("ok", "verified")):
            raise RuntimeError("discovery did not produce a verified real artifact")
        (data_dir / "discovery.json").write_text(
            json.dumps(discovery, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        actions.append("discovery")
        state = BootstrapState.DISCOVERED
    else:
        discovery = _read(data_dir / "discovery.json") or {}

    if state == BootstrapState.DISCOVERED:
        data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
        selection = make_selection_artifact(
            str(discovery.get("run_id")),
            discovery.get("candidates", []),
            allowed_org=str(data_cfg.get("org", "openeurollm")),
            allowed_licenses=data_cfg.get("allowed_licenses", []),
            allowed_sources=data_cfg.get("allowed_sources", []),
        )
        selection["status"] = "ok"
        selection["mode"] = "real"
        # status and mode are part of the authoritative hash.
        body = {key: value for key, value in selection.items() if key != "artifact_hash"}
        from .data_pipeline import canonical_json, sha256_bytes

        selection["artifact_hash"] = sha256_bytes(canonical_json(body))
        (data_dir / "selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not selection["sources"]:
            raise RuntimeError("discovery contains no deterministically selectable sources")
        result = dict(prepare(discovery, selection))
        if not _verified(result, ("trainable", "ok")):
            raise RuntimeError("data preparation did not produce a verified trainable manifest")
        actions.append("data_prepare")
        state = BootstrapState.DATA_READY

    if state == BootstrapState.DATA_READY:
        result = dict(baseline())
        if not (
            result.get("status") in {"ok", "pass"}
            and result.get("mode") == "real"
            and int(result.get("technical_error_count", 0)) == 0
        ):
            raise RuntimeError("baseline did not produce a successful real dev evaluation")
        actions.append("dev_baseline")

    final_state = derive_state(root)
    if final_state != BootstrapState.RESEARCH_READY:
        raise RuntimeError(f"bootstrap stopped at {final_state.name}")
    return {"status": "ok", "state": final_state.name, "actions": actions}
