import json
import shutil
import subprocess

import pytest

from boldt_posttrain.merge import build_candidates, mergekit_config, run_merge_round
from boldt_posttrain.training import make_peft_config


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
