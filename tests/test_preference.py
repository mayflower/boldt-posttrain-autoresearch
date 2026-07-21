from pathlib import Path

import pytest
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from boldt_posttrain.policy import load_policy
from boldt_posttrain.preference import (
    PreferenceError,
    preference_dataset,
    train_preference_adapter,
    validate_preference_rows,
)
from tests.test_training_preflight import experiment
from tests.tiny_model import build_tiny_model


def preference_settings() -> dict:
    return {
        "method": "dpo",
        "loss_type": "sigmoid",
        "beta": 0.1,
        "rpo_alpha": 0.0,
        "length_ratio_max": 3.0,
        "max_prompt_length": 32,
        "max_completion_length": 16,
    }


def rows() -> list[dict[str, str]]:
    return [
        {"prompt": "Hallo", "chosen": "Buch", "rejected": "Regen"},
        {"prompt": "Regen", "chosen": "Regen fällt", "rejected": "Buch ."},
    ]


def test_preference_rows_reject_empty_identical_and_oversized(tmp_path: Path):
    model_path = build_tiny_model(tmp_path / "tiny")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    settings = preference_settings()
    with pytest.raises(PreferenceError, match="empty or identical"):
        validate_preference_rows(
            [{"prompt": "Hallo", "chosen": "Buch", "rejected": "Buch"}],
            tokenizer,
            settings,
        )
    with pytest.raises(PreferenceError, match="prompt exceeds"):
        validate_preference_rows(
            [{"prompt": "Hallo " * 40, "chosen": "Buch", "rejected": "Regen"}],
            tokenizer,
            settings,
        )


def test_kto_conversion_contains_both_label_classes():
    dataset = preference_dataset(rows(), "kto")
    assert len(dataset) == 4
    assert set(dataset["label"]) == {True, False}


@pytest.mark.parametrize("method", ["dpo", "kto", "orpo"])
def test_real_preference_method_trains_and_reloads_adapter(
    method: str, tmp_path: Path, monkeypatch
):
    import boldt_posttrain.preference as preference

    model_path = build_tiny_model(tmp_path / "tiny")
    state_root = tmp_path / "repo/outputs/posttrain"
    monkeypatch.setattr(preference, "OUTPUTS", state_root)
    training = experiment()
    training["max_steps"] = 1
    if method == "kto":
        training["per_device_batch_size"] = 2
    result = train_preference_adapter(
        method=method,
        model_source=str(model_path),
        revision=None,
        rows=rows(),
        output_root=state_root / "checkpoints",
        policy=load_policy(),
        training=training,
        preference=preference_settings(),
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
    )
    assert result["status"] == "succeeded"
    assert result["method"] == method
    checkpoint = Path(result["checkpoint"])
    assert (checkpoint / "adapter_model.safetensors").is_file()
    base = AutoModelForCausalLM.from_pretrained(model_path)
    adapter = PeftModel.from_pretrained(base, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter(**tokenizer("Hallo", return_tensors="pt"))
