"""Secure training preflight: device binding, warmup translation, budget outcomes."""

import pytest

from boldt_posttrain.training import warmup_steps_from_ratio


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


def test_reload_verified_budget_stop_is_successful():
    from boldt_posttrain.preference import (
        _completion_outcome as preference_completion_outcome,
    )
    from boldt_posttrain.training import (
        DeadlineCallback as SecureDeadlineCallback,
        _completion_outcome,
    )

    callback = SecureDeadlineCallback(float("inf"))
    callback.exhausted = True
    assert _completion_outcome(callback) == ("succeeded", "budget_limit")
    assert preference_completion_outcome(callback) == ("succeeded", "budget_limit")


def test_secure_qlora_honors_explicit_cuda_device(monkeypatch):
    import torch
    import transformers
    import peft
    import boldt_posttrain.training as secure_training

    captured = {}
    model = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(secure_training, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: captured.update(kwargs) or model,
    )
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", lambda value, **kwargs: value)
    policy = type(
        "Policy",
        (),
        {
            "document": {
                "training": {
                    "qlora": {
                        "quant_type": "nf4",
                        "double_quant": True,
                        "compute_dtype": "bfloat16",
                    }
                }
            }
        },
    )()

    loaded, _tokenizer = secure_training.create_model_and_tokenizer(
        "/tmp/model",
        revision=None,
        qlora=True,
        gradient_checkpointing=False,
        policy=policy,
        device="cuda:1",
    )

    assert loaded is model
    assert captured["device_map"] == {"": 1}


def test_warmup_ratio_maps_to_warmup_steps_and_rejects_out_of_range():
    assert warmup_steps_from_ratio(0.03, 1000) == 0.03
    assert warmup_steps_from_ratio(1.0, 50) == 50
    for invalid in (-0.1, 1.5):
        with pytest.raises(ValueError, match="range"):
            warmup_steps_from_ratio(invalid, 10)
    with pytest.raises(ValueError, match="positive max_steps"):
        warmup_steps_from_ratio(1.0, -1)
