import time
import builtins

import pytest

from boldt_posttrain.preference import DeadlineCallback
from boldt_posttrain.training import (
    align_quantized_output_head_input,
    _assistant_mask_metrics,
    _cpt_peft_and_optimizer,
    evaluation_interval,
    make_peft_config,
    validate_assistant_supervision,
    validate_liger,
)


def test_evaluation_interval_is_short_run_safe_and_capped():
    assert evaluation_interval(1) == 1
    assert evaluation_interval(95) == 9
    assert evaluation_interval(10_000) == 100


def test_quantized_output_head_alignment_preserves_fp32_head():
    import torch

    class QuantizedModel:
        is_loaded_in_4bit = True

        def __init__(self):
            self.head = torch.nn.Linear(4, 3, bias=False, dtype=torch.float32)

        def get_output_embeddings(self):
            return self.head

    model = QuantizedModel()
    align_quantized_output_head_input(model)
    align_quantized_output_head_input(model)
    output = model.head(torch.ones((1, 4), dtype=torch.bfloat16))
    assert output.dtype == torch.float32
    assert len(model.head._forward_pre_hooks) == 1


def test_deadline_callback_stops_at_reserved_boundary():
    callback = DeadlineCallback(time.monotonic() + 10, reserve_seconds=20)
    control = type("Control", (), {"should_training_stop": False})()
    callback.on_step_end(None, None, control)
    assert control.should_training_stop
    assert callback.stop_reason == "budget_limit"


class BrokenTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [1, 2, 3]}


def test_missing_assistant_mask_fails_before_training():
    rows = [{"messages": [{"role": "user", "content": "F"}, {"role": "assistant", "content": "A"}]}]
    with pytest.raises(ValueError, match="assistant token mask"):
        validate_assistant_supervision(rows, [], BrokenTokenizer(), context_length=8)


class TruncatedTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1:
            return {"input_ids": [1, 2], "assistant_masks": [0, 0]}
        return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 1]}


def test_truncation_that_removes_supervision_fails():
    rows = [{"messages": [{"role": "user", "content": "F"}, {"role": "assistant", "content": "A"}]}]
    with pytest.raises(ValueError, match="after truncation"):
        _assistant_mask_metrics(rows, TruncatedTokenizer(), context_length=2)


class OverbroadTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs):
        count = len(messages)
        return {"input_ids": list(range(count)), "assistant_masks": [1] * count}


def test_user_tokens_marked_as_assistant_fail():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "F"},
                {"role": "assistant", "content": "A"},
            ]
        }
    ]
    with pytest.raises(ValueError, match="non-assistant"):
        _assistant_mask_metrics(rows, OverbroadTokenizer(), context_length=8)


def test_rslora_reaches_real_peft_config():
    config = make_peft_config(
        {
            "lora_r": 4,
            "lora_alpha": 8,
            "target_modules": ["q_proj"],
            "use_rslora": True,
        }
    )
    assert config.use_rslora is True


def test_liger_missing_or_unsupported_fails_before_training(monkeypatch):
    with pytest.raises(ValueError, match="does not support"):
        validate_liger(True, model_type="unsupported-fixture")
    original_import = builtins.__import__

    def without_liger(name, *args, **kwargs):
        if name.startswith("liger_kernel"):
            raise ImportError("fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_liger)
    with pytest.raises(RuntimeError, match="perf extra"):
        validate_liger(True, model_type="llama")


def test_cpt_saved_modules_get_separate_learning_rates(tiny_model_dir):
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    training = {
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "learning_rate": 1e-3,
        "method": "lora",
    }
    wrapped, peft_config, optimizer, modules, _tied = _cpt_peft_and_optimizer(
        model,
        training,
        {
            "train_embeddings": True,
            "train_lm_head": True,
            "module_learning_rate_multiplier": 0.1,
        },
    )
    assert wrapped is not model
    assert peft_config is None
    assert set(modules) == {"embed_tokens", "lm_head"}
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]


def test_cpt_saved_modules_reject_qlora(tiny_model_dir):
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    with pytest.raises(ValueError, match="QLoRA"):
        _cpt_peft_and_optimizer(
            model,
            {
                "lora_r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj"],
                "learning_rate": 1e-3,
                "method": "qlora",
            },
            {
                "train_embeddings": True,
                "train_lm_head": False,
                "module_learning_rate_multiplier": 0.1,
            },
        )
