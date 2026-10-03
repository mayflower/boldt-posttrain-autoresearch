import json
from pathlib import Path

import pytest

from boldt_posttrain.preference import (
    normalize_preference_row,
    render_probe,
    train_preference,
    validate_preference_rows,
)
from boldt_posttrain.training import make_peft_config


ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_all_preference_rows_are_conversational_and_probe_uses_template():
    row = normalize_preference_row({"prompt": "P", "chosen": "C", "rejected": "R"})
    assert row["prompt"] == [{"role": "user", "content": "P"}]
    rendered = render_probe(Tokenizer(), row)
    assert rendered["chosen"] == "user:P|assistant:C"


def test_multiturn_validation_uses_template_token_limits(tiny_model_dir):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model_dir)
    row = normalize_preference_row(
        {
            "prompt": [
                {"role": "system", "content": "eins"},
                {"role": "user", "content": "Frage eins"},
                {"role": "assistant", "content": "Antwort eins"},
                {"role": "user", "content": "Frage zwei"},
            ],
            "chosen": [{"role": "assistant", "content": "Antwort richtig"}],
            "rejected": [{"role": "assistant", "content": "Antwort falsch"}],
        }
    )
    diagnostics = validate_preference_rows(
        [row],
        tokenizer,
        max_prompt_length=32,
        max_completion_length=16,
        context_length=64,
        length_ratio_max=3.0,
    )
    assert [message["role"] for message in row["prompt"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert diagnostics["prompt_tokens"]["count"] == 1
    with pytest.raises(ValueError, match="prompt would be truncated"):
        validate_preference_rows(
            [row],
            tokenizer,
            max_prompt_length=1,
            max_completion_length=16,
            context_length=64,
            length_ratio_max=3.0,
        )


@pytest.mark.parametrize("method", ["dpo", "kto", "orpo"])
def test_conversational_fixture_trains_each_preference_method_one_step(
    tmp_path, tiny_model_dir, method
):
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    rows = [
        normalize_preference_row(
            {"prompt": "Frage eins", "chosen": "Antwort richtig", "rejected": "Antwort falsch"}
        ),
        normalize_preference_row(
            {"prompt": "Frage zwei", "chosen": "Antwort zwei", "rejected": "Antwort drei"}
        ),
    ]
    result = train_preference(
        method=method,
        model=str(tiny_model_dir),
        tokenizer=transformers.AutoTokenizer.from_pretrained(tiny_model_dir),
        dataset=datasets.Dataset.from_list(rows),
        output_dir=tmp_path / method,
        training_args={
            "max_steps": 1,
            "per_device_train_batch_size": 2,
            "learning_rate": 1e-3,
            "max_length": 32,
            "gradient_checkpointing": False,
            "save_strategy": "no",
            "report_to": "none",
            "use_cpu": True,
            "disable_tqdm": True,
            "logging_steps": 1,
        },
        peft_config=make_peft_config(
            {
                "lora_r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
                "lora_init": "default",
            }
        ),
    )
    assert result.global_step == 1


def test_dpo_validation_and_sft_loss_use_pinned_api(tmp_path, tiny_model_dir):
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    rows = [
        normalize_preference_row(
            {"prompt": "Frage eins", "chosen": "Antwort richtig", "rejected": "Antwort falsch"}
        ),
        normalize_preference_row(
            {"prompt": "Frage zwei", "chosen": "Antwort zwei", "rejected": "Antwort drei"}
        ),
    ]
    result = train_preference(
        method="dpo",
        model=str(tiny_model_dir),
        tokenizer=transformers.AutoTokenizer.from_pretrained(tiny_model_dir),
        dataset=datasets.Dataset.from_list(rows),
        eval_dataset=datasets.Dataset.from_list(rows),
        output_dir=tmp_path / "dpo-eval",
        training_args={
            "max_steps": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 1e-3,
            "max_length": 32,
            "gradient_checkpointing": False,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",
            "use_cpu": True,
            "disable_tqdm": True,
        },
        preference_config={
            "loss_type": "sigmoid",
            "sft_loss_weight": 0.25,
            "beta": 0.1,
            "max_prompt_length": 16,
            "max_completion_length": 16,
        },
        peft_config=make_peft_config(
            {
                "lora_r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
    )
    assert result.trainer.eval_dataset is not None
    assert result.trainer.args.loss_type == ["sigmoid", "sft"]
    assert result.trainer.args.loss_weights == [1.0, 0.25]
    assert result.trainer.state.best_metric is not None


@pytest.mark.parametrize("method", ["kto", "orpo"])
def test_sft_loss_weight_is_rejected_for_unsupported_methods(method):
    with pytest.raises(ValueError, match="only for DPO"):
        train_preference(
            method=method,
            model=None,
            tokenizer=None,
            dataset=None,
            output_dir=Path("unused"),
            training_args={},
            preference_config={"sft_loss_weight": 0.1},
        )


@pytest.mark.parametrize("method", ["kto", "orpo"])
def test_kto_and_orpo_use_validation_and_select_best_checkpoint(tmp_path, tiny_model_dir, method):
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    rows = [
        normalize_preference_row(
            {"prompt": "Frage eins", "chosen": "Antwort richtig", "rejected": "Antwort falsch"}
        ),
        normalize_preference_row(
            {"prompt": "Frage zwei", "chosen": "Antwort zwei", "rejected": "Antwort drei"}
        ),
    ]
    result = train_preference(
        method=method,
        model=str(tiny_model_dir),
        tokenizer=transformers.AutoTokenizer.from_pretrained(tiny_model_dir),
        dataset=datasets.Dataset.from_list(rows),
        eval_dataset=datasets.Dataset.from_list(rows),
        output_dir=tmp_path / f"{method}-eval",
        training_args={
            "max_steps": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 1e-3,
            "max_length": 32,
            "gradient_checkpointing": False,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",
            "use_cpu": True,
            "disable_tqdm": True,
        },
        preference_config={
            "sft_loss_weight": 0.0,
            "max_prompt_length": 16,
            "max_completion_length": 16,
        },
        peft_config=make_peft_config(
            {
                "lora_r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
    )
    assert result.trainer.eval_dataset is not None
    assert result.trainer.state.best_metric is not None


@pytest.mark.parametrize("method", ["dpo", "kto", "orpo"])
def test_sequence_bound_is_prompt_plus_completion_not_trl_default(tmp_path, tiny_model_dir, method):
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    rows = [
        normalize_preference_row(
            {"prompt": "Frage eins", "chosen": "Antwort richtig", "rejected": "Antwort falsch"}
        ),
        normalize_preference_row(
            {"prompt": "Frage zwei", "chosen": "Antwort zwei", "rejected": "Antwort drei"}
        ),
    ]
    common = {
        "max_steps": 1,
        "per_device_train_batch_size": 2,
        "learning_rate": 1e-3,
        "gradient_checkpointing": False,
        "save_strategy": "no",
        "report_to": "none",
        "use_cpu": True,
        "disable_tqdm": True,
    }
    peft = make_peft_config(
        {
            "lora_r": 4,
            "lora_alpha": 8,
            "target_modules": ["q_proj", "v_proj"],
            "lora_init": "default",
        }
    )
    bounds = {"max_prompt_length": 1000, "max_completion_length": 1000}
    for index, (caller_bound, expected) in enumerate(((None, 2000), (40, 40))):
        result = train_preference(
            method=method,
            model=str(tiny_model_dir),
            tokenizer=transformers.AutoTokenizer.from_pretrained(tiny_model_dir),
            dataset=datasets.Dataset.from_list(rows),
            output_dir=tmp_path / f"{method}-{index}",
            training_args=common
            if caller_bound is None
            else {**common, "max_length": caller_bound},
            preference_config=bounds,
            peft_config=peft,
        )
        # Never TRL's silent 1024-token default: the gates admit prompt + completion.
        assert result.trainer.args.max_length == expected


def _secure_preference_call(monkeypatch, tmp_path, method, rpo_alpha):
    from types import SimpleNamespace

    from boldt_posttrain.secure_compat import preference as secure

    captured = {}

    class Stop(Exception):
        pass

    def config_class(**kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.setattr(
        secure, "create_model_and_tokenizer", lambda *a, **k: (SimpleNamespace(), None)
    )
    for name in ("validate_tokenizer", "validate_target_modules"):
        monkeypatch.setattr(secure, name, lambda *a, **k: None)
    monkeypatch.setattr(secure, "collect_model_metadata", lambda *a, **k: {})
    monkeypatch.setattr(secure, "validate_preference_rows", lambda *a, **k: {})
    monkeypatch.setattr(secure, "response_suppression_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(secure, "preference_dataset", lambda *a, **k: [])
    monkeypatch.setattr(secure, "_trainer", lambda m: (None, config_class))
    document = json.loads((ROOT / "configs/posttrain/secure-current.json").read_text())
    preference = {**document["preference"], "method": method, "rpo_alpha": rpo_alpha}
    training = {**document["training"], "per_device_batch_size": 2}
    call = dict(
        method=method,
        model_source="seed",
        revision=None,
        rows=[{"prompt": "p", "chosen": "a", "rejected": "b"}],
        output_root=tmp_path / "checkpoints",
        policy=None,
        training=training,
        preference=preference,
        target_modules=["q_proj"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=1,
        repository_root=tmp_path,
    )
    return secure, Stop, captured, call


def test_secure_dpo_turns_rpo_alpha_into_weighted_sft_loss(monkeypatch, tmp_path):
    secure, stop, captured, call = _secure_preference_call(monkeypatch, tmp_path, "dpo", 0.5)
    with pytest.raises(stop):
        secure.train_preference_adapter(**call)
    assert captured["loss_type"] == ["sigmoid", "sft"]
    assert captured["loss_weights"] == [1.0, 0.5]
    secure, stop, captured, call = _secure_preference_call(monkeypatch, tmp_path, "dpo", 0.0)
    with pytest.raises(stop):
        secure.train_preference_adapter(**call)
    assert captured["loss_type"] == ["sigmoid"] and "loss_weights" not in captured


@pytest.mark.parametrize("method", ["kto", "orpo"])
def test_secure_rpo_alpha_is_rejected_for_non_dpo_methods(monkeypatch, tmp_path, method):
    secure, _, _, call = _secure_preference_call(monkeypatch, tmp_path, method, 0.5)
    monkeypatch.setattr(
        secure, "create_model_and_tokenizer", lambda *a, **k: pytest.fail("must reject first")
    )
    with pytest.raises(secure.PreferenceError, match="applies only to DPO"):
        secure.train_preference_adapter(**call)
