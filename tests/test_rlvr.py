import time

import pytest

from boldt_posttrain.data_pipeline import FastTextLanguageIdentifier
from boldt_posttrain.rlvr import make_reward_callable, train_rlvr, validate_rlvr_row


def test_rlvr_schema_and_callable_reward():
    row = validate_rlvr_row(
        {
            "prompt": [{"role": "user", "content": "1+1"}],
            "task_type": "numeric",
            "ground_truth": {"value": 2},
            "reward_version": 1,
            "source": {},
            "license": "generated",
            "content_id": "x",
            "leakage_clean": True,
            "training_usable": True,
        }
    )
    reward = make_reward_callable(
        weights={"numeric": 1},
        clamp=[-1, 2],
        language_id=FastTextLanguageIdentifier(predictor=lambda _text: ("de", 1.0)),
    )
    assert reward(
        completions=["2"], task_type=[row["task_type"]], ground_truth=[row["ground_truth"]]
    ) == [1]


def test_reward_failure_propagates_instead_of_becoming_zero():
    reward = make_reward_callable(
        weights={"numeric": 1},
        clamp=[-1, 2],
        language_id=FastTextLanguageIdentifier(predictor=lambda _text: ("de", 1.0)),
    )
    with pytest.raises(KeyError):
        reward(completions=["2"], task_type=["numeric"], ground_truth=[{}])


def test_tiny_rloo_performs_real_optimizer_step_and_reload(tmp_path, tiny_model_dir):
    datasets = pytest.importorskip("datasets")
    language = FastTextLanguageIdentifier(predictor=lambda _text: ("de", 1.0))
    rows = [
        validate_rlvr_row(
            {
                "prompt": [{"role": "user", "content": "eins plus eins"}],
                "task_type": "numeric",
                "ground_truth": {"value": 2},
                "reward_version": 1,
                "source": {"kind": "procedural"},
                "license": "generated",
                "content_id": f"numeric-{index}",
                "leakage_clean": True,
                "training_usable": True,
            }
        )
        for index in range(2)
    ]
    result = train_rlvr(
        model_ref=str(tiny_model_dir),
        dataset=datasets.Dataset.from_list(rows),
        output_dir=tmp_path / "rloo",
        config={
            "training": {
                "max_steps": 1,
                "learning_rate": 1e-3,
                "lora_r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
                "lora_init": "default",
            },
            "rlvr": {
                "batch_size": 2,
                "num_generations": 2,
                "max_completion_length": 4,
                "gradient_accumulation_steps": 1,
            },
        },
        policy={"reward_weights": {"numeric": 1.0}, "reward_clamp": [-1.0, 2.0]},
        device="cpu",
        deadline=time.monotonic() + 120,
        language_id=language,
    )
    assert result["status"] == "ok"
    assert result["metrics"]["trainable_parameters"] > 0
    assert result["metrics"]["checkpoint_bytes"] > 0
