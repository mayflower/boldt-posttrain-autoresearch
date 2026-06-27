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

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "outputs" / "posttrain" / "evals"
FRONTIER = ROOT / "outputs" / "posttrain" / "frontier.json"

# German-helpfulness dimensions used for the quick aggregate ranking.
DIMS = ["german_instruction", "format_following", "reasoning_core", "longcontext"]


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
    return sorted(d.name for d in evals_dir.iterdir()
                  if d.is_dir() and (d / "summary.json").exists())


def build_frontier(evals_dir: Optional[Path] = None) -> Dict[str, Any]:
    evals_dir = Path(evals_dir) if evals_dir else EVALS
    candidates = []
    for label in _labels(evals_dir):
        doc = _summary(label, evals_dir)
        metrics = doc.get("metrics", {}) if isinstance(doc.get("metrics"), dict) else {}
        candidates.append({
            "label": label,
            "mode": doc.get("mode"),
            "status": doc.get("status"),
            "real": doc.get("mode") == "real" and not doc.get("scale_disclaimer"),
            "aggregate": aggregate(metrics),
            "dims": {d: _num(metrics.get(d)) for d in DIMS},
        })
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
        "note": ("German-helpfulness aggregate over saved eval summaries only. Dry-run candidates "
                 "are listed but never rank as frontier-best. Merging is most promising when the "
                 "per-dimension leaders are DIFFERENT checkpoints sharing the warm-start basin."),
    }


def current_frontier() -> Dict[str, Any]:
    if FRONTIER.exists():
        try:
            return json.loads(FRONTIER.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
