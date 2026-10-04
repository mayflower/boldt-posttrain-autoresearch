"""run_experiment's pass/promote/reject branches, with the heavy stages mocked."""

from types import SimpleNamespace

import pytest

from boldt_posttrain import loop, provenance
from boldt_posttrain.frontier import FrontierNotImproved
from boldt_posttrain.resolver import ResolvedModelRef


def _resolved() -> ResolvedModelRef:
    return ResolvedModelRef(
        kind="peft_adapter",
        requested="cand",
        base_model={"repo_id": "seed", "revision": "r"},
        artifact={"path": "outputs/posttrain/checkpoints/cand", "role": "adapter_checkpoint"},
        tokenizer_sha256="t",
        chat_template_sha256="c",
        model_config_sha256="m",
        architecture="LlamaForCausalLM",
        source_run_id="train-sft-20260101T000000.000000Z-0123456789abcdef",
    )


def _wire(monkeypatch, tmp_path, *, score_status: str, baseline_seconds: float = 60.0):
    """Mock every heavy stage so only run_experiment's branching runs."""
    calls = {"promote": 0, "lever_deadline": None}
    monkeypatch.setattr(loop, "load_policy", lambda: object())
    monkeypatch.setattr(
        loop.config_module, "load_experiment", lambda _p: type("C", (), {"document": {}})()
    )
    monkeypatch.setattr(provenance, "resolve_base_ref", lambda ref, root: ref)
    monkeypatch.setattr(
        loop,
        "load_baseline",
        lambda *a, **k: SimpleNamespace(
            run_card={"run_id": "baseline-1", "duration_seconds": baseline_seconds}
        ),
    )
    monkeypatch.setattr(loop, "verify_data_manifest", lambda *a, **k: {"status": "trainable"})

    def _lever(*a, deadline, **k):
        calls["lever_deadline"] = deadline
        return {"run_id": _resolved().source_run_id, "status": "succeeded"}

    monkeypatch.setattr(loop, "_execute_lever", _lever)
    monkeypatch.setattr(loop, "resolve_model", lambda *a, **k: _resolved())
    monkeypatch.setattr(loop, "_publish_evaluation", lambda *a, **k: {"run_id": "eval-1"})
    monkeypatch.setattr(
        loop,
        "create_score",
        lambda *a, **k: {"score_run_id": "score-1", "status": score_status, "score": 0.5},
    )
    monkeypatch.setattr(loop, "_integrity_check", lambda *a, **k: {"status": "pass"})
    monkeypatch.setattr(loop, "current_frontier_hash", lambda *a, **k: None)

    def _promote(*a, **k):
        calls["promote"] += 1
        return {"promotion_id": "p1", "status": "promoted"}

    monkeypatch.setattr(loop, "promote_candidate", _promote)
    return calls


def test_passing_candidate_with_promote_is_promoted(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, score_status="passed")
    verdict, code = loop.run_experiment(
        config_path=tmp_path / "exp.json",
        base_ref="HEAD",
        budget_minutes=90,
        promote=True,
        allow_checkpoints=True,
        allow_gpu=True,
        outputs_root=tmp_path / "outputs",
        repository_root=tmp_path,
    )
    assert code == 0
    assert verdict["status"] == "promoted"
    assert verdict["disposition"] == "promoted"
    assert calls["promote"] == 1


def test_passing_candidate_without_promote_stops_at_succeeded(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, score_status="passed")
    verdict, code = loop.run_experiment(
        config_path=tmp_path / "exp.json",
        base_ref="HEAD",
        budget_minutes=90,
        promote=False,
        allow_checkpoints=True,
        allow_gpu=True,
        outputs_root=tmp_path / "outputs",
        repository_root=tmp_path,
    )
    assert code == 0
    assert verdict["status"] == "succeeded"
    assert calls["promote"] == 0


def test_rejected_candidate_never_promotes(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, score_status="rejected")
    verdict, code = loop.run_experiment(
        config_path=tmp_path / "exp.json",
        base_ref="HEAD",
        budget_minutes=90,
        promote=True,
        allow_checkpoints=True,
        allow_gpu=True,
        outputs_root=tmp_path / "outputs",
        repository_root=tmp_path,
    )
    assert code == 1
    assert verdict["status"] == "rejected"
    assert verdict["disposition"] == "rejected"
    assert calls["promote"] == 0


def _run(tmp_path, *, budget_minutes=90, promote=True):
    return loop.run_experiment(
        config_path=tmp_path / "exp.json",
        base_ref="HEAD",
        budget_minutes=budget_minutes,
        promote=promote,
        allow_checkpoints=True,
        allow_gpu=True,
        outputs_root=tmp_path / "outputs",
        repository_root=tmp_path,
    )


def test_candidate_that_does_not_beat_the_champion_is_a_normal_rejection(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, score_status="passed")

    def _not_better(*a, **k):
        raise FrontierNotImproved(0.5, {"candidate_run_id": "champion", "score": 0.7})

    monkeypatch.setattr(loop, "promote_candidate", _not_better)
    verdict, code = _run(tmp_path)
    assert code == 1
    assert verdict["status"] == "rejected" and verdict["disposition"] == "not_promoted"
    assert "does not beat the champion" in verdict["stages"]["promotion"]["reason"]
    assert verdict["error"] is None


def test_training_stops_early_enough_to_leave_time_for_evaluation(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, score_status="rejected", baseline_seconds=600.0)
    started = loop.time.monotonic()
    _run(tmp_path, budget_minutes=90)
    reserve = loop.evaluation_reserve_seconds(600.0)
    assert reserve == pytest.approx(1020.0)
    assert calls["lever_deadline"] == pytest.approx(started + 90 * 60 - reserve, abs=5)


def test_budget_without_room_for_training_fails_before_training(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, score_status="passed", baseline_seconds=1500.0)
    verdict, code = _run(tmp_path, budget_minutes=45)
    assert code == 4 and calls["lever_deadline"] is None
    assert "raise --budget-minutes" in verdict["error"]
