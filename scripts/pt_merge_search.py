#!/usr/bin/env python3
"""Merge search over complementary, compatible specialist checkpoints.

Dry mode scans ``outputs/posttrain/runs/*/run_card.json`` for eligible training runs (same
warm-start basin), enumerates a merge MATRIX (candidate pairs × configured methods) with verdict
``needs_eval``, and writes ``merge/<merge_id>/merge_matrix.json``. Real mode executes mergekit and
then selects one candidate through proxy-to-dev evaluation.

    python scripts/pt_merge_search.py --config configs/posttrain/current.json \
        --runs outputs/posttrain/runs --out outputs/posttrain/merge --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402
from boldt_posttrain import provenance as prov  # noqa: E402
from boldt_posttrain import recipe  # noqa: E402
from boldt_posttrain.merge import mergekit_config  # noqa: E402
from boldt_posttrain.merge import run_merge_round  # noqa: E402
from boldt_posttrain.evaluation import run_real_evaluation  # noqa: E402
from boldt_posttrain.frontier import aggregate, verified_merge_inputs  # noqa: E402


def _eligible(runs_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Training runs that are merge-eligible: a run card with a train_* run_type + a base model."""
    out: List[Dict[str, Any]] = []
    if not runs_dir.exists():
        return out
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        card_p = d / "run_card.json"
        if not card_p.exists():
            continue
        try:
            card = json.loads(card_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(card.get("run_type", "")).startswith("train_"):
            artifacts = card.get("output_artifacts", [])
            checkpoint = next((value for value in artifacts if pathlib.Path(value).is_dir()), None)
            out.append(
                {
                    "run_id": card.get("run_id", d.name),
                    "base_model": card.get("model"),
                    "run_type": card.get("run_type"),
                    "checkpoint": checkpoint,
                }
            )
    return out


def build_matrix(eligible: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    """Enumerate candidate merges: pairs sharing a base model × each configured method."""
    candidates: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(eligible, 2):
        if a["base_model"] != b["base_model"]:
            continue  # only merge descendants of the same warm-start basin
        for method in methods:
            candidates.append(
                {
                    "run_id": f"{a['run_id']}+{b['run_id']}::{method}",
                    "parents": [a["run_id"], b["run_id"]],
                    "method": method,
                    "parameters": {},
                    "eval_summary": None,
                    "verdict": "needs_eval",
                }
            )
    return candidates


def _frontier_eligible(path: pathlib.Path, base_model: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "run_id": str(item["run_id"]),
            "base_model": base_model,
            "run_type": "verified_specialist_frontier",
            "checkpoint": str(item["model"]),
            "frontier": item["frontier"],
        }
        for item in verified_merge_inputs(document)
    ]


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--runs", default=str(ROOT / "outputs/posttrain/runs"))
    ap.add_argument("--frontier", default=str(ROOT / "outputs/posttrain/specialist-frontiers.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/posttrain/merge"))
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--merge-device", default="cpu")
    ap.add_argument("--device", default="cuda:0", help="explicit evaluation device")
    args = ap.parse_args(argv)
    dry = not args.real

    cfg = cfgmod.resolve_config(pathlib.Path(args.config))
    methods = cfg.get("merge", {}).get("methods", ["linear", "slerp", "ties", "dare_ties"])
    base_model = str(cfg.get("merge", {}).get("base_model"))
    eligible = _eligible(pathlib.Path(args.runs))
    eligible.extend(_frontier_eligible(pathlib.Path(args.frontier), base_model))
    eligible = list({str(item["run_id"]): item for item in eligible}.values())
    candidate_limit = int(cfg.get("merge", {}).get("max_candidates", 12))
    candidates = build_matrix(eligible, methods)[:candidate_limit]

    merge_id = f"merge-{'dry' if dry else 'real'}-{prov.stamp()}"
    out_dir = pathlib.Path(args.out) / merge_id

    matrix: Dict[str, Any] = {
        "merge_id": merge_id,
        "mode": "dry_run" if dry else "real",
        "base_model": base_model,
        "methods": methods,
        "n_eligible": len(eligible),
        "eligible": eligible,
        "candidates": candidates,
        "note": "merge only same-basin descendants; verdict needs_eval until evaluated/scored.",
    }
    if dry:
        matrix["status"] = "ok"
        matrix["scale_disclaimer"] = "dry-run plumbing only — no merge was performed"
    else:
        if args.merge_device != "cpu" and not args.merge_device.startswith("cuda:"):
            matrix["status"] = "failed"
            matrix["message"] = "--merge-device must be cpu or an explicit cuda:N"
        elif args.merge_device.startswith("cuda:") and not args.allow_gpu:
            matrix["status"] = "failed"
            matrix["message"] = "CUDA merge requires --allow-gpu"
        else:
            matrix["status"] = "ok"
            for index, candidate in enumerate(candidates):
                left, right = [
                    next(item for item in eligible if item["run_id"] == run_id)
                    for run_id in candidate["parents"]
                ]
                if not left.get("checkpoint") or not right.get("checkpoint"):
                    candidate.update(status="failed", technical_error="checkpoint_missing")
                    matrix["status"] = "failed"
                    continue
                candidate_dir = out_dir / f"candidate-{index:03d}"
                config_path = candidate_dir / "merge.yml"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    json.dumps(
                        mergekit_config(
                            method=candidate["method"],
                            base_model=matrix["base_model"],
                            models=[left["checkpoint"], right["checkpoint"]],
                            dtype=cfg.get("merge", {}).get("dtype", "bfloat16"),
                            tokenizer_source=cfg.get("merge", {}).get("tokenizer_source", "union"),
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                command = [
                    "mergekit-yaml",
                    str(config_path),
                    str(candidate_dir / "model"),
                    "--device",
                    args.merge_device,
                    "--lora-merge-cache",
                    str(candidate_dir / "lora-cache"),
                ]
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    candidate.update(
                        status="failed",
                        technical_error="merge_error",
                        stderr=completed.stderr[-2000:],
                    )
                    matrix["status"] = "failed"
                else:
                    candidate.update(status="ok", checkpoint=str(candidate_dir / "model"))
            if candidates and any(candidate.get("status") == "ok" for candidate in candidates):
                successful = [
                    candidate for candidate in candidates if candidate.get("status") == "ok"
                ]
                suite = ROOT / cfg.get("eval", {}).get("dev_suite", "data/eval/dev.json")

                def evaluate(candidate, profile):
                    started = time.monotonic()
                    summary = run_real_evaluation(
                        model_ref=candidate["checkpoint"],
                        suite_path=suite,
                        profile=profile,
                        device=args.device,
                        output_dir=out_dir / candidate["run_id"] / profile,
                        config=cfg,
                        deadline=time.monotonic() + 90 * 60,
                    )
                    return {
                        "status": summary["status"],
                        "technical_error_count": summary["technical_error_count"],
                        "hard_gates_passed": summary["status"] == "ok"
                        and all(summary.get("hard_gates", {}).values()),
                        "proxy_score": aggregate(summary["metrics"]),
                        "gpu_seconds": time.monotonic() - started,
                        "summary": summary,
                    }

                round_result = run_merge_round(
                    candidates=successful,
                    proxy_evaluate=lambda candidate: evaluate(candidate, "proxy"),
                    dev_evaluate=lambda candidate: evaluate(candidate, "dev"),
                )
                matrix["selection"] = round_result
                matrix["status"] = round_result["status"]
            elif not candidates:
                matrix["status"] = "rejected"

    recipe.write_json(out_dir / "merge_matrix.json", matrix)
    if args.format == "markdown":
        print(f"# Merge search — {matrix['status']} ({matrix['mode']})\n")
        print(f"- eligible specialists: {len(eligible)}  ·  candidate merges: {len(candidates)}")
        print(f"- methods: {', '.join(methods)}  ·  out: `{out_dir / 'merge_matrix.json'}`")
        if len(eligible) < 2:
            print("\n_Fewer than 2 eligible specialists — train more branches before merging._")
        if matrix.get("message"):
            print(f"- {matrix['message']}")
    else:
        print(
            json.dumps(
                {
                    "status": matrix["status"],
                    "merge_id": merge_id,
                    "n_eligible": len(eligible),
                    "n_candidates": len(candidates),
                    "out": str(out_dir / "merge_matrix.json"),
                },
                ensure_ascii=False,
            )
        )
    return 0 if matrix["status"] == "ok" else (1 if matrix["status"] == "rejected" else 4)


if __name__ == "__main__":
    raise SystemExit(main())
