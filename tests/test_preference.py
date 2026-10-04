"""Secure preference training: sequence bounds and the RPO term."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _preference_call(monkeypatch, tmp_path, method, rpo_alpha):
    from types import SimpleNamespace

    from boldt_posttrain import preference

    captured = {}

    class Stop(Exception):
        pass

    def config_class(**kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.setattr(
        preference, "create_model_and_tokenizer", lambda *a, **k: (SimpleNamespace(), None)
    )
    for name in ("validate_tokenizer", "validate_target_modules"):
        monkeypatch.setattr(preference, name, lambda *a, **k: None)
    monkeypatch.setattr(preference, "collect_model_metadata", lambda *a, **k: {})
    monkeypatch.setattr(preference, "validate_preference_rows", lambda *a, **k: {})
    monkeypatch.setattr(preference, "response_suppression_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(preference, "preference_dataset", lambda *a, **k: [])
    monkeypatch.setattr(preference, "_trainer", lambda m: (None, config_class))
    document = json.loads((ROOT / "configs/posttrain/secure-current.json").read_text())
    settings = {**document["preference"], "method": method, "rpo_alpha": rpo_alpha}
    training = {**document["training"], "per_device_batch_size": 2}
    call = dict(
        method=method,
        model_source="seed",
        revision=None,
        rows=[{"prompt": "p", "chosen": "a", "rejected": "b"}],
        output_root=tmp_path / "checkpoints",
        policy=None,
        training=training,
        preference=settings,
        target_modules=["q_proj"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=1,
        repository_root=tmp_path,
    )
    return preference, Stop, captured, call


def test_dpo_turns_rpo_alpha_into_weighted_sft_loss(monkeypatch, tmp_path):
    preference, stop, captured, call = _preference_call(monkeypatch, tmp_path, "dpo", 0.5)
    with pytest.raises(stop):
        preference.train_preference_adapter(**call)
    assert captured["loss_type"] == ["sigmoid", "sft"]
    assert captured["loss_weights"] == [1.0, 0.5]
    preference, stop, captured, call = _preference_call(monkeypatch, tmp_path, "dpo", 0.0)
    with pytest.raises(stop):
        preference.train_preference_adapter(**call)
    assert captured["loss_type"] == ["sigmoid"] and "loss_weights" not in captured


@pytest.mark.parametrize("method", ["kto", "orpo"])
def test_rpo_alpha_is_rejected_for_non_dpo_methods(monkeypatch, tmp_path, method):
    preference, _, _, call = _preference_call(monkeypatch, tmp_path, method, 0.5)
    monkeypatch.setattr(
        preference, "create_model_and_tokenizer", lambda *a, **k: pytest.fail("must reject first")
    )
    with pytest.raises(preference.PreferenceError, match="applies only to DPO"):
        preference.train_preference_adapter(**call)


@pytest.mark.parametrize("method", ["dpo", "kto", "orpo"])
def test_sequence_bound_is_prompt_plus_completion_capped_by_context(monkeypatch, tmp_path, method):
    preference, stop, captured, call = _preference_call(monkeypatch, tmp_path, method, 0.0)
    call["preference"].update(max_prompt_length=1000, max_completion_length=1000)
    call["training"]["context_length"] = 16384
    with pytest.raises(stop):
        preference.train_preference_adapter(**call)
    # Never TRL's 1024-token default: the length gates admit prompt + completion.
    assert captured["max_length"] == 2000
    call["training"]["context_length"] = 1500
    with pytest.raises(stop):
        preference.train_preference_adapter(**call)
    assert captured["max_length"] == 1500
