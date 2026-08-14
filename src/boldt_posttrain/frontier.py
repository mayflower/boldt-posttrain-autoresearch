"""Read-only frontier view over saved eval summaries (pure stdlib).

A metric is only what was saved: this scans ``outputs/posttrain/evals/<label>/summary.json`` and
reports each candidate's German-helpfulness aggregate, ranks them, and surfaces per-dimension
leaders (complementary specialists worth MERGING). It never trains and never claims beyond the
saved summaries. ``scripts/pt_frontier_status.py`` is the CLI; ``scripts/pt_promote.py`` writes the
authoritative ``frontier.json`` only when the protected gate passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_pipeline import canonical_json, sha256_bytes, verify_hashed_artifact

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "outputs" / "posttrain" / "evals"
FRONTIER = ROOT / "outputs" / "posttrain" / "frontier.json"

# German-helpfulness dimensions used for the quick aggregate ranking.
DIMS = ["german_instruction", "format_following", "reasoning_core", "longcontext"]
SPECIALIST_DIMENSIONS = {
    "reasoning": "reasoning_core",
    "coding": "coding",
    "format": "format_following",
    "longcontext": "longcontext",
    "safety": "safety",
}


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _summary(label: str, evals_dir: Path) -> Dict[str, Any]:
    p = evals_dir / label / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate(metrics: Dict[str, Any]) -> Optional[float]:
    vals = [_num(metrics.get(d)) for d in DIMS]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _labels(evals_dir: Path) -> List[str]:
    if not evals_dir.exists():
        return []
    return sorted(
        d.name for d in evals_dir.iterdir() if d.is_dir() and (d / "summary.json").exists()
    )


def build_frontier(evals_dir: Optional[Path] = None) -> Dict[str, Any]:
    evals_dir = Path(evals_dir) if evals_dir else EVALS
    candidates = []
    for label in _labels(evals_dir):
        doc = _summary(label, evals_dir)
        metrics = doc.get("metrics", {}) if isinstance(doc.get("metrics"), dict) else {}
        candidates.append(
            {
                "label": label,
                "mode": doc.get("mode"),
                "status": doc.get("status"),
                "real": doc.get("mode") == "real" and not doc.get("scale_disclaimer"),
                "aggregate": aggregate(metrics),
                "dims": {d: _num(metrics.get(d)) for d in DIMS},
            }
        )
    real = [c for c in candidates if c["real"] and c["aggregate"] is not None]
    real.sort(key=lambda c: c["aggregate"], reverse=True)

    leaders: Dict[str, Any] = {}
    for d in DIMS:
        best, lbl = None, None
        for c in candidates:
            v = c["dims"].get(d)
            if v is not None and (best is None or v > best):
                best, lbl = v, c["label"]
        leaders[d] = {"label": lbl, "score": best}
    complementary = sorted({v["label"] for v in leaders.values() if v["label"]})

    return {
        "n_candidates": len(candidates),
        "n_real": len(real),
        "frontier_best": (real[0] if real else None),
        "per_dimension_leaders": leaders,
        "complementary_merge_inputs": complementary,
        "candidates": candidates,
        "note": (
            "German-helpfulness aggregate over saved eval summaries only. Dry-run candidates "
            "are listed but never rank as frontier-best. Merging is most promising when the "
            "per-dimension leaders are DIFFERENT checkpoints sharing the warm-start basin."
        ),
    }


def current_frontier() -> Dict[str, Any]:
    if FRONTIER.exists():
        try:
            return json.loads(FRONTIER.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def specialist_eligible(summary: Dict[str, Any]) -> bool:
    metrics = summary.get("metrics", {})
    leakage = metrics.get("leakage", {}) if isinstance(metrics, dict) else {}
    license_block = metrics.get("license", {}) if isinstance(metrics, dict) else {}
    return (
        verify_hashed_artifact(summary)
        and summary.get("mode") == "real"
        and summary.get("status") in {"ok", "pass"}
        and summary.get("technical_error_count") == 0
        and leakage.get("status") in {"clean", "verified_clean"}
        and leakage.get("hits", 0) == 0
        and license_block.get("usable") is True
        and summary.get("hard_gates", {}).get("language") is True
        and summary.get("hard_gates", {}).get("safety") is True
        and summary.get("hard_gates", {}).get("format") is True
    )


def update_specialist_frontiers(
    candidates: List[Dict[str, Any]], current: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Update five explicit specialist frontiers without an N-dimensional framework."""
    existing = (current or {}).get("specialists", {})
    specialists = dict(existing) if isinstance(existing, dict) else {}
    for name, dimension in SPECIALIST_DIMENSIONS.items():
        best = specialists.get(name)
        best_score = _num(best.get("score")) if isinstance(best, dict) else None
        for summary in candidates:
            if not specialist_eligible(summary):
                continue
            score = _num(summary.get("metrics", {}).get(dimension))
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best = {
                    "run_id": summary.get("run_id"),
                    "model": summary.get("model"),
                    "dimension": dimension,
                    "score": score,
                    "eval_artifact_hash": summary.get("artifact_hash"),
                }
        if best is not None:
            specialists[name] = best
    body = {
        "schema_version": 1,
        "general": (current or {}).get("general"),
        "specialists": specialists,
    }
    return {**body, "artifact_hash": sha256_bytes(canonical_json(body))}


def verified_merge_inputs(frontier_document: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not verify_hashed_artifact(frontier_document):
        raise ValueError("frontier artifact hash is invalid")
    values = []
    for name, item in sorted(frontier_document.get("specialists", {}).items()):
        if isinstance(item, dict) and item.get("model"):
            values.append({"frontier": name, **item})
    return values


from .secure_compat.frontier import (  # noqa: E402, F401
    FrontierError,
    _integrity_check,
    current_frontier_hash,
    frontier_status,
    promote_candidate,
    verify_frontier,
)
