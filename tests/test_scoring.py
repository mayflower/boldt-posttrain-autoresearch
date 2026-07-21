import json
from pathlib import Path

import pytest

from boldt_posttrain.scoring import (
    ScoringError,
    create_score,
    load_candidate_score,
    verify_evaluation,
)
from tests.artifact_chain import complete_chain


def test_complete_real_artifact_chain_scores_and_recomputes(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    result = create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    assert result["status"] == "passed"
    assert result["score"] > 0
    artifact, card = load_candidate_score(
        chain["candidate_run_id"],
        chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    assert artifact["gates"]["positive_score"] is True
    assert card["status"] == "succeeded"
    assert artifact["statistics"]["german_instruction"]["n"] == 60


def test_missing_or_nonreal_mode_fails_scoring(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    path = chain["outputs"] / "evals" / chain["candidate_eval_run_id"] / "summary.json"
    summary = json.loads(path.read_text())
    summary.pop("mode")
    path.write_text(json.dumps(summary))
    with pytest.raises(ScoringError):
        create_score(
            chain["candidate_eval_run_id"],
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
        )


def test_every_policy_metric_is_required(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    path = chain["outputs"] / "evals" / chain["candidate_eval_run_id"] / "summary.json"
    summary = json.loads(path.read_text())
    del summary["metrics"]["safety"]
    path.write_text(json.dumps(summary))
    with pytest.raises(ScoringError):
        verify_evaluation(
            path.parent,
            chain["policy"],
            outputs_root=chain["outputs"],
            baseline=False,
            repository_root=chain["repository"],
        )


def test_nonfinite_metric_fails(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    path = chain["outputs"] / "evals" / chain["candidate_eval_run_id"] / "summary.json"
    summary = json.loads(path.read_text())
    summary["metrics"]["safety"] = float("nan")
    path.write_text(json.dumps(summary))
    with pytest.raises(ScoringError, match="non-finite"):
        verify_evaluation(
            path.parent,
            chain["policy"],
            outputs_root=chain["outputs"],
            baseline=False,
            repository_root=chain["repository"],
        )


def test_negative_score_cannot_pass(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch, improvement=-0.1)
    result = create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    assert result["status"] == "rejected"
    assert result["score"] < 0


def test_checkpoint_mutation_invalidates_stored_score(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    checkpoint = chain["outputs"] / "checkpoints" / chain["candidate_run_id"]
    (checkpoint / "adapter_model.safetensors").write_bytes(b"manipulated")
    with pytest.raises(Exception, match="verification|hash|candidate"):
        load_candidate_score(
            chain["candidate_run_id"],
            chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
        )


def test_free_summary_is_not_a_scoreable_chain(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    fake = chain["outputs"] / "evals/fake"
    fake.mkdir()
    (fake / "summary.json").write_text(json.dumps({"status": "succeeded", "score": 100}))
    with pytest.raises(ScoringError):
        create_score(
            "fake",
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
        )
