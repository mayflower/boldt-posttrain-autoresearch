#!/usr/bin/env python3
"""Append one auditable row per scored trial to ``outputs/posttrain/results.tsv`` (pure stdlib).

Stable columns; header written only when the file is new; old rows are never rewritten. Reads a run
directory containing ``summary.json`` (eval) or ``run_card.json``/``metrics.json``, plus an optional
``score.json``. This is the loop's append-only audit log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "posttrain" / "results.tsv"

COLUMNS = [
    "timestamp_utc", "commit", "label", "mode", "status", "score",
    "german_instruction", "format_following", "reasoning_core", "longcontext",
    "english_bleed_rate", "empty_output_rate", "refusal_rate", "safety",
    "leakage_status", "license_status", "config_path", "notes",
]


def _dig(d: Any, *path: str) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _clean(v: Any) -> str:
    if v is None:
        return ""
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _load(run_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load the primary doc from a run dir: prefer summary.json, then run_card.json/metrics.json."""
    for name in ("summary.json", "run_card.json", "metrics.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def build_row(doc: Dict[str, Any], score_doc: Optional[Dict[str, Any]],
              status_override: Optional[str], notes: Optional[str],
              timestamp_utc: str) -> Dict[str, str]:
    m = doc.get("metrics", {}) if isinstance(doc.get("metrics"), dict) else {}
    status = status_override or (score_doc or {}).get("status") or doc.get("status")
    row = {
        "timestamp_utc": timestamp_utc,
        "commit": doc.get("commit") or _dig(doc, "git", "commit"),
        "label": doc.get("label") or doc.get("run_id"),
        "mode": doc.get("mode"),
        "status": status,
        "score": (score_doc or {}).get("score"),
        "german_instruction": m.get("german_instruction"),
        "format_following": m.get("format_following"),
        "reasoning_core": m.get("reasoning_core"),
        "longcontext": m.get("longcontext"),
        "english_bleed_rate": m.get("english_bleed_rate"),
        "empty_output_rate": m.get("empty_output_rate"),
        "refusal_rate": m.get("refusal_rate"),
        "safety": m.get("safety"),
        "leakage_status": _dig(m, "leakage", "status"),
        "license_status": _dig(m, "license", "status"),
        "config_path": doc.get("config_path") or doc.get("suite"),
        "notes": notes,
    }
    return {k: _clean(row.get(k)) for k in COLUMNS}


def append_row(results_path: pathlib.Path, row: Dict[str, str]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results_path.exists()
    with open(results_path, "a", encoding="utf-8") as fh:
        if new_file:
            fh.write("\t".join(COLUMNS) + "\n")
        fh.write("\t".join(row.get(c, "") for c in COLUMNS) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir with summary.json/run_card.json/metrics.json")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--status", default=None,
                    help="override disposition: keep|discard|crash|invalid_leakage|needs_real")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run)
    doc = _load(run_dir)
    if doc is None:
        print(json.dumps({"appended": False,
                          "error": f"no summary.json/run_card.json/metrics.json under {run_dir}"},
                         ensure_ascii=False))
        return 2
    score_path = run_dir / "score.json"
    score_doc = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else None

    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    row = build_row(doc, score_doc, args.status, args.notes, ts)
    append_row(pathlib.Path(args.results), row)
    print(json.dumps({"appended": True, "label": row["label"], "status": row["status"],
                      "score": row["score"], "results": args.results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
