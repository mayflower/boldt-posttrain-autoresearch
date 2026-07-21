import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from boldt_posttrain.policy import load_policy
from boldt_posttrain.training import (
    DeadlineCallback,
    TrainingError,
    collect_model_metadata,
    create_model_and_tokenizer,
    train_adapter,
    validate_target_modules,
)
from tests.tiny_model import build_tiny_model


def experiment() -> dict:
    return {
        "specialist": "general-de",
        "method": "lora",
        "learning_rate": 1e-4,
        "num_train_epochs": 1.0,
        "max_steps": 1,
        "warmup_ratio": 0.0,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "context_length": 64,
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["c_attn"],
        "seed": 42,
        "packing": False,
        "gradient_checkpointing": False,
        "assistant_only_loss": False,
        "quantization": "nf4",
    }


def test_target_modules_are_validated_against_real_model(tmp_path: Path):
    model_path = build_tiny_model(tmp_path / "tiny")
    model, _ = create_model_and_tokenizer(
        str(model_path),
        revision=None,
        qlora=False,
        gradient_checkpointing=False,
        policy=load_policy(),
        device="cpu",
    )
    validate_target_modules(model, ["c_attn"])
    with pytest.raises(TrainingError, match="do not exist"):
        validate_target_modules(model, ["q_proj"])


def test_budget_deadline_stops_at_step_boundary():
    callback = DeadlineCallback(time.monotonic() - 1)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_step_end(None, None, control)
    assert control.should_training_stop is True
    assert callback.exhausted is True


def test_checkpoint_write_requires_explicit_permission(tmp_path: Path):
    from datasets import Dataset

    model_path = build_tiny_model(tmp_path / "tiny")
    checkpoint_root = tmp_path / "outputs/posttrain/checkpoints"
    with pytest.raises(TrainingError, match="explicit permission"):
        train_adapter(
            kind="sft",
            model_source=str(model_path),
            revision=None,
            dataset=Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "Hallo"},
                            {"role": "assistant", "content": "Hallo"},
                        ]
                    }
                ]
            ),
            output_root=checkpoint_root,
            policy=load_policy(),
            experiment=experiment(),
            target_modules=["c_attn"],
            device="cpu",
            qlora=False,
            allow_checkpoints=False,
            budget_minutes=1,
        )
    assert not checkpoint_root.exists()


def test_protected_seed_requires_exact_revision_before_download():
    policy = load_policy()
    with pytest.raises(TrainingError, match="exact policy revision"):
        collect_model_metadata(policy.seed_model["repo_id"], None, object(), policy)
