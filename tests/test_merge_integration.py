import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from boldt_posttrain.evaluation import finalize_summary
from boldt_posttrain.frontier import update_specialist_frontiers
from boldt_posttrain.merge import build_candidates, mergekit_config, run_merge_round
from boldt_posttrain.training import make_peft_config


def _merge_script():
    path = Path(__file__).resolve().parents[1] / "scripts/pt_merge_search.py"
    spec = importlib.util.spec_from_file_location("pt_merge_search_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiple_merge_candidates_choose_one_full_eval():
    parents = [{"run_id": name, "base_model": "seed"} for name in ("a", "b", "c")]
    candidates = build_candidates(parents, ["linear", "ties"], limit=5)
    dev_calls = []

    def proxy(candidate):
        return {
            "proxy_score": float(candidate["run_id"].startswith("a")),
            "gpu_seconds": 2,
            "technical_error_count": 0,
            "hard_gates_passed": True,
        }

    def dev(candidate):
        dev_calls.append(candidate["run_id"])
        return {"status": "ok", "technical_error_count": 0}

    result = run_merge_round(candidates=candidates, proxy_evaluate=proxy, dev_evaluate=dev)
    assert result["status"] == "ok"
    assert len(candidates) == 5
    assert len(dev_calls) == 1


def test_verified_specialist_frontiers_feed_merge_matrix(tmp_path):
    summary = finalize_summary(
        {
            "run_id": "reasoning-run",
            "model": str(tmp_path / "adapter"),
            "mode": "real",
            "status": "ok",
            "technical_error_count": 0,
            "hard_gates": {"language": True, "safety": True, "format": True},
            "metrics": {
                "reasoning_core": 0.9,
                "leakage": {"status": "clean", "hits": 0},
                "license": {"usable": True},
            },
        }
    )
    frontier = update_specialist_frontiers([summary])
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps(frontier), encoding="utf-8")
    eligible = _merge_script()._frontier_eligible(path, "seed")
    assert eligible == [
        {
            "run_id": "reasoning-run",
            "base_model": "seed",
            "run_type": "verified_specialist_frontier",
            "checkpoint": str(tmp_path / "adapter"),
            "frontier": "reasoning",
        }
    ]
    matrix = _merge_script().build_matrix(
        eligible
        + [
            {
                "run_id": "format-run",
                "base_model": "seed",
                "checkpoint": str(tmp_path / "format-adapter"),
            }
        ],
        ["linear"],
    )
    assert matrix[0]["run_id"] == "reasoning-run+format-run::linear"


def test_all_merge_configs_validate_against_locked_mergekit():
    mergekit = pytest.importorskip("mergekit.config")
    for method in ("linear", "slerp", "ties", "dare_ties"):
        mergekit.MergeConfiguration.model_validate(
            mergekit_config(
                method=method,
                base_model="seed",
                models=["left", "right"],
                dtype="bfloat16",
            )
        )


@pytest.mark.parametrize("method", ["linear", "slerp", "ties", "dare_ties"])
def test_locked_mergekit_executes_real_tiny_adapter_merge(tmp_path, tiny_model_dir, method):
    executable = shutil.which("mergekit-yaml")
    if not executable:
        pytest.skip("mergekit console executable is not installed")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    adapters = []
    for index in range(2):
        model = transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir)
        model = peft.get_peft_model(
            model,
            make_peft_config(
                {
                    "lora_r": 4,
                    "lora_alpha": 8,
                    "target_modules": ["q_proj", "v_proj"],
                    "lora_init": "default",
                }
            ),
        )
        adapter = tmp_path / f"adapter-{index}"
        model.save_pretrained(adapter)
        adapters.append(str(adapter))
    config = mergekit_config(
        method=method,
        base_model=str(tiny_model_dir),
        models=adapters,
        dtype="float32",
    )
    config_path = tmp_path / "merge.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "merged"
    completed = subprocess.run(
        [
            executable,
            str(config_path),
            str(output),
            "--device",
            "cpu",
            "--lora-merge-cache",
            str(tmp_path / "lora-cache"),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(output)
    assert reloaded.config.vocab_size == 17
