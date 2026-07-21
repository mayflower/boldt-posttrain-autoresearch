import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from boldt_posttrain.policy import load_policy
from boldt_posttrain.training import train_adapter
from tests.test_training_preflight import experiment
from tests.tiny_model import build_tiny_model


GPU_48GB_AVAILABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 47 * 1024**3
)


@pytest.mark.skipif(not GPU_48GB_AVAILABLE, reason="requires one visible 48-GB CUDA GPU")
def test_real_two_step_qlora_on_48gb_gpu(tmp_path: Path):
    model_path = build_tiny_model(tmp_path / "tiny")
    state_root = tmp_path / "repo/outputs/posttrain"
    training = experiment()
    training["max_steps"] = 2
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hallo"},
                    {"role": "assistant", "content": "Buch"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Regen"},
                    {"role": "assistant", "content": "Regen fällt"},
                ]
            },
        ]
    )
    result = train_adapter(
        kind="sft",
        model_source=str(model_path),
        revision=None,
        dataset=dataset,
        output_root=state_root / "checkpoints",
        policy=load_policy(),
        experiment=training,
        target_modules=["c_attn"],
        device="cuda:0",
        qlora=True,
        allow_checkpoints=True,
        budget_minutes=2,
    )
    assert result["status"] == "succeeded"
    assert result["metrics"]["steps_completed"] == 2
    assert result["metrics"]["peak_gpu_memory_bytes"] > 0
    card = json.loads((state_root / "runs" / result["run_id"] / "run_card.json").read_text())
    assert card["hardware"]["gpus"][0]["vram_bytes"] >= 47 * 1024**3
    assert card["hardware"]["gpus"][0]["compute_capability"] == "8.6"
