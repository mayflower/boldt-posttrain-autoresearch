import time

import pytest

from boldt_posttrain.verified_rl import (
    make_grpo_config,
    math_accuracy_reward,
    parse_gold_solution,
    train_verified_grpo,
    validate_verified_math_rows,
)


def config(tiny_model_dir):
    return {
        "training": {
            "base_model": str(tiny_model_dir),
            "context_length": 64,
            "max_steps": 1,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.0,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "seed": 17,
            "lora_r": 4,
            "lora_alpha": 8,
            "target_modules": ["q_proj", "v_proj"],
            "use_rslora": True,
            "use_liger_kernel": False,
        },
        "verified_rl": {
            "enabled": True,
            "reward_profile": "math_accuracy",
            "num_generations": 2,
            "batch_size": 2,
            "max_prompt_length": 32,
            "max_completion_length": 16,
            "temperature": 1.0,
            "beta": 0.0,
        },
    }


def test_math_verify_accepts_equivalent_fraction_and_rejects_wrong_answer():
    assert math_accuracy_reward(
        [[{"role": "assistant", "content": r"\boxed{\frac{1}{3}}"}]],
        [r"\frac{1}{3}"],
    ) == [1.0]
    assert math_accuracy_reward(["2"], ["3"]) == [0.0]


def test_unparseable_gold_is_rejected():
    with pytest.raises(ValueError, match="could not parse"):
        parse_gold_solution("not math")


def test_conversational_prompt_and_gold_are_validated(tiny_model_dir):
    from transformers import AutoTokenizer

    row = {
        "content_id": "row-1",
        "prompt": [{"role": "user", "content": "1 + 1"}],
        "solution": "2",
    }
    result = validate_verified_math_rows(
        [row],
        AutoTokenizer.from_pretrained(tiny_model_dir),
        max_prompt_length=32,
    )
    assert result["rows"] == 1
    assert row["prompt"] == [{"role": "user", "content": "1 + 1"}]


def test_grpo_config_gets_effective_values_and_rslora_is_separate(tmp_path, tiny_model_dir):
    cfg = config(tiny_model_dir)
    args = make_grpo_config(cfg, tmp_path, device="cpu", has_validation=True)
    assert args.num_generations == 2
    assert args.max_prompt_length == 32
    assert args.max_completion_length == 16
    assert args.temperature == 1.0
    assert args.beta == 0.0
    assert args.use_liger_kernel is False
    assert args.metric_for_best_model == "eval_rewards/math_accuracy_reward/mean"


def test_invalid_generation_batch_ratio_fails(tmp_path, tiny_model_dir):
    cfg = config(tiny_model_dir)
    cfg["verified_rl"]["batch_size"] = 3
    with pytest.raises(ValueError, match="divisible"):
        make_grpo_config(cfg, tmp_path, device="cpu", has_validation=True)


def test_disabled_grpo_fails_before_model_initialization(tmp_path, tiny_model_dir):
    cfg = config(tiny_model_dir)
    cfg["verified_rl"]["enabled"] = False
    with pytest.raises(ValueError, match="disabled"):
        train_verified_grpo(
            model_ref="must-not-load",
            train_dataset=[{}],
            eval_dataset=[{}],
            output_dir=tmp_path,
            config=cfg,
            device="cuda:0",
            deadline=time.monotonic() + 60,
        )


def test_grpo_rejects_unsupported_training_method_before_model_initialization(
    tmp_path, tiny_model_dir
):
    cfg = config(tiny_model_dir)
    cfg["training"]["method"] = "full"
    with pytest.raises(ValueError, match="only explicit lora or qlora"):
        train_verified_grpo(
            model_ref="must-not-load",
            train_dataset=[{}],
            eval_dataset=[{}],
            output_dir=tmp_path,
            config=cfg,
            device="cuda:0",
            deadline=time.monotonic() + 60,
        )


def test_qlora_grpo_rejects_cpu_before_model_initialization(monkeypatch, tmp_path, tiny_model_dir):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    cfg = config(tiny_model_dir)
    cfg["training"]["method"] = "qlora"
    rows = Dataset.from_list([{"prompt": [{"role": "user", "content": "1 + 1"}], "solution": "2"}])

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("model loading must not begin for CPU QLoRA")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match="QLoRA requires an explicit CUDA device"):
        train_verified_grpo(
            model_ref=str(tiny_model_dir),
            train_dataset=rows,
            eval_dataset=rows,
            output_dir=tmp_path,
            config=cfg,
            device="cpu",
            deadline=time.monotonic() + 60,
        )


def test_real_lora_grpo_budget_stop_validates_saves_and_reloads(
    monkeypatch, tmp_path, tiny_model_dir
):
    from datasets import Dataset

    from boldt_posttrain.preference import DeadlineCallback

    cfg = config(tiny_model_dir)
    cfg["training"].update({"method": "lora", "max_steps": 2, "use_rslora": False})
    train_rows = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": "1 + 1"}], "solution": "2"},
            {"prompt": [{"role": "user", "content": "1 + 2"}], "solution": "3"},
        ]
    )
    eval_rows = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": "2 + 1"}], "solution": "3"},
            {"prompt": [{"role": "user", "content": "2 + 2"}], "solution": "4"},
        ]
    )

    def stop_after_first_step(self, args, state, control, **_kwargs):
        self.stop_reason = "budget_limit"
        control.should_evaluate = True
        control.should_save = True
        control.should_training_stop = True
        return control

    monkeypatch.setattr(DeadlineCallback, "on_step_end", stop_after_first_step)
    result = train_verified_grpo(
        model_ref=str(tiny_model_dir),
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        output_dir=tmp_path / "run",
        config=cfg,
        device="cpu",
        deadline=time.monotonic() + 120,
    )

    assert result["status"] == "succeeded"
    assert result["metrics"]["steps"] == 1
    assert result["metrics"]["stop_reason"] == "budget_limit"
    assert result["metrics"]["validation_evaluations"] >= 1
    assert result["metrics"]["best_step"] >= 1
    assert result["metrics"]["reward_mean"] == 0.0
    assert result["metrics"]["reward_std"] == 0.0
    assert result["metrics"]["frac_reward_zero_std"] == 1.0
    assert result["metrics"]["completion_clipping_fraction"] is not None
    assert result["metrics"]["mean_completion_length"] is not None
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").is_file()
    assert not (tmp_path / "run" / "trainer").exists()
