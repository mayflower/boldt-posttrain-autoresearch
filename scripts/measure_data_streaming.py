#!/usr/bin/env python3
"""Materialize the required 100k-row fixture and print deterministic RSS/shard evidence."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain.data_pipeline import materialize_streaming  # noqa: E402


def rows(count: int):
    for index in range(count):
        digest = hashlib.sha256(str(index).encode()).hexdigest()
        yield {"instruction": f"Aufgabe {digest[:24]}", "answer": f"Antwort {digest[24:48]}"}


def main() -> int:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with tempfile.TemporaryDirectory() as directory:
        source = {
            "dataset": "rss-fixture",
            "source_group": "general",
            "schema": "sft",
            "license": "apache-2.0",
            "rows": rows(100_000),
        }
        manifest = materialize_streaming([source], Path(directory) / "data")
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        evidence = {
            "status": "ok",
            "input_rows": 100_000,
            "written_rows": manifest["row_counts"].get("written", 0),
            "shards": len(manifest["shards"]),
            "peak_rss_kib": after,
            "peak_rss_delta_kib": max(0, after - before),
            "manifest_hash": manifest["artifact_hash"],
        }
        print(json.dumps(evidence, sort_keys=True))
        if evidence["written_rows"] < 95_000 or evidence["shards"] < 3:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
