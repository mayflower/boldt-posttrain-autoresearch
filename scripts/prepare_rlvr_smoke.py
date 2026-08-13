#!/usr/bin/env python3
"""Create a verified procedural RLVR smoke manifest from the current decontamination corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain.data_pipeline import materialize_streaming, verify_hashed_artifact  # noqa: E402


def rows(count: int):
    for index in range(count):
        left = 100 + index
        right = 7 + index % 5
        yield {
            "prompt": [
                {
                    "role": "user",
                    "content": f"Berechne {left} plus {right}. Antworte nur mit der Zahl.",
                }
            ],
            "task_type": "numeric",
            "ground_truth": {"value": left + right},
            "reward_version": 1,
            "source": {"kind": "procedural_gpu_smoke", "index": index},
            "license": "generated",
            "content_id": f"gpu-smoke-numeric-{index:04d}",
            "leakage_clean": True,
            "training_usable": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    parser.add_argument("--decontamination", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()
    cfg = cfgmod.resolve_config(Path(args.config))
    corpus = json.loads(Path(args.decontamination).read_text(encoding="utf-8"))
    if not verify_hashed_artifact(corpus) or corpus.get("policy_hash") != cfg.get("policy_hash"):
        raise ValueError("RLVR smoke data requires the current verified decontamination corpus")
    manifest = materialize_streaming(
        [
            {
                "dataset": "procedural-gpu-smoke",
                "source_group": "reasoning",
                "schema": "rlvr",
                "license": "generated",
                "rows": rows(args.rows),
            }
        ],
        Path(args.out),
        decontamination_corpus=corpus,
        policy_hash=cfg["policy_hash"],
    )
    print(json.dumps({"status": "ok", "manifest_hash": manifest["artifact_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
