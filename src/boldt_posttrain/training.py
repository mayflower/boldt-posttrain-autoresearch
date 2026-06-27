"""Shared training-lever orchestration for specialist / preference / CPT branches (pure stdlib).

All three training scripts share the same auditable skeleton: resolve config, persist provenance,
and either (dry) write a training PLAN + unmeasured run card, or (real) verify the human GPU gate,
the optional ML stack, and a clean data manifest — then fail closed because the concrete trainer
is not implemented in this scaffold (it must be added per the script contracts, never faked).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import provenance as prov
from . import recipe

CONTRACT = "docs/posttrain-script-contracts.md"
ROOT = Path(__file__).resolve().parents[2]

# Which optional preference loss / training knobs to surface in the dry-run plan, per kind.
_RUN_TYPE = {"specialist": "train_specialist", "preference": "train_preference",
             "cpt": "train_cpt"}


def _training_plan(cfg: Dict[str, Any], kind: str, specialist: str) -> Dict[str, Any]:
    tr = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    plan = {
        "kind": kind,
        "specialist": specialist,
        "base_model": tr.get("base_model"),
        "method": tr.get("method"),
        "learning_rate": tr.get("learning_rate"),
        "num_train_epochs": tr.get("num_train_epochs"),
        "max_steps": tr.get("max_steps"),
        "context_length": tr.get("context_length"),
        "lora_r": tr.get("lora_r"),
        "lora_alpha": tr.get("lora_alpha"),
        "target_modules": tr.get("target_modules"),
    }
    if kind == "preference":
        plan["preference"] = cfg.get("preference", {})
    return plan


def _manifest_clean(data_dir: Path) -> Dict[str, Any]:
    """Fail-closed check that prepared data is trainable: manifest present + leakage verified clean."""
    manifest = data_dir / "manifest.json"
    leakage = data_dir / "leakage_report.json"
    if not manifest.exists():
        return {"clean": False, "reason": f"no data manifest at {manifest} — run /pt-data real"}
    if not leakage.exists():
        return {"clean": False, "reason": f"no leakage report at {leakage} — fails closed"}
    try:
        lk = json.loads(leakage.read_text(encoding="utf-8"))
    except Exception:
        return {"clean": False, "reason": "leakage_report.json unparseable — fails closed"}
    status = str(lk.get("status", "")).lower()
    if status not in ("clean", "verified_clean", "ok"):
        return {"clean": False, "reason": f"leakage status '{status}' is not verified clean"}
    return {"clean": True, "reason": "manifest present and leakage verified clean"}


def run_training_trial(*, cfg: Dict[str, Any], kind: str, specialist: str, out_root: Path,
                       budget_minutes: int, argv: List[str], dry_run: bool, allow_gpu: bool,
                       allow_checkpoints: bool, data_dir: Path,
                       config_errors: List[str]) -> int:
    run_id = f"{specialist}-{kind}-{'dry' if dry_run else 'real'}-{prov.stamp()}"
    out_dir = Path(out_root) / run_id
    git = recipe.persist_inputs(out_dir, cfg, argv)
    command = "python " + " ".join(argv)
    plan = _training_plan(cfg, kind, specialist)

    def finish(metrics: Dict[str, Any], status: str, extra: Dict[str, Any]) -> int:
        metrics_doc = {"run_id": run_id, "status": status, "mode": "dry_run" if dry_run else "real",
                       "budget_minutes": budget_minutes, "git": git, "training_plan": plan,
                       "metrics": metrics, **extra}
        recipe.write_json(out_dir / "metrics.json", metrics_doc)
        card = prov.new_run_card(run_id, _RUN_TYPE[kind], command,
                                 model=plan.get("base_model"), metrics=metrics,
                                 data_manifest=str(data_dir / "manifest.json"),
                                 input_artifacts=[str(p) for p in (data_dir / "manifest.json",)],
                                 output_artifacts=[str(out_dir / "metrics.json")],
                                 notes=extra.get("message", f"{kind} {specialist} ({metrics_doc['mode']})"))
        prov.write_run_card(card, out_dir)
        print(json.dumps({"status": status, "mode": metrics_doc["mode"], "run_id": run_id,
                          "out": str(out_dir), **{k: extra[k] for k in ("message",) if k in extra}},
                         ensure_ascii=False))
        return 0 if status in ("ok", "pass") else 2

    if config_errors:
        return finish(recipe.metrics_skeleton(cfg), "fail",
                      {"message": "config invalid: " + "; ".join(config_errors)})

    if dry_run:
        return finish(recipe.metrics_skeleton(cfg), "ok",
                      {"scale_disclaimer": "dry-run plumbing only — no checkpoint, no metrics",
                       "message": f"planned {kind} '{specialist}'; pass --real --allow-gpu to train"})

    # --- real path: human gate, ML stack, clean data, then fail closed (no faked trainer) ---
    if not allow_gpu:
        return finish(recipe.metrics_skeleton(cfg), "fail",
                      {"message": "--real requires --allow-gpu (human hardware gate)"})
    stack_err = recipe.require_real_stack()
    if stack_err:
        return finish(recipe.metrics_skeleton(cfg), "fail", {"message": stack_err})
    clean = _manifest_clean(data_dir)
    if not clean["clean"]:
        return finish(recipe.metrics_skeleton(cfg), "fail",
                      {"message": "data not trainable: " + clean["reason"]})
    ni = recipe.real_not_implemented(f"train_{kind}", CONTRACT)
    return finish(recipe.metrics_skeleton(cfg), "fail",
                  {"missing_real_implementation": ni["missing_real_implementation"],
                   "message": ni["message"]})
