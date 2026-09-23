"""run_experiment orchestration: the pass/promote/reject branches.

The individual stages are covered elsewhere (promote_candidate in test_frontier,
create_score in the scoring tests, the eval publisher by a real baseline run).
What no test exercised is loop.run_experiment's own decision logic -- that a
passing score with promote=True calls promote_candidate and reports "promoted",
that promote=False stops at "succeeded", and that a rejected score never promotes.
These mock the heavy stages and assert only that wiring.
"""

from boldt_posttrain import loop, provenance
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


def _wire(monkeypatch, tmp_path, *, score_status: str):
    """Mock every heavy stage so only run_experiment's branching runs."""
    calls = {"promote": 0}
    monkeypatch.setattr(loop, "load_policy", lambda: object())
    monkeypatch.setattr(
        loop.config_module, "load_experiment", lambda _p: type("C", (), {"document": {}})()
    )
    monkeypatch.setattr(provenance, "resolve_base_ref", lambda ref, root: ref)
    monkeypatch.setattr(loop, "load_baseline", lambda *a, **k: object())
    monkeypatch.setattr(loop, "verify_data_manifest", lambda *a, **k: {"status": "trainable"})
    monkeypatch.setattr(
        loop,
        "_execute_lever",
        lambda *a, **k: {"run_id": _resolved().source_run_id, "status": "succeeded"},
    )
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
