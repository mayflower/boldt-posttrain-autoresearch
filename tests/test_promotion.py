import json
from pathlib import Path

import pytest

from boldt_posttrain.frontier import FrontierError, promote_candidate
from boldt_posttrain.scoring import create_score
from tests.artifact_chain import complete_chain


def passing_integrity(base_ref, repository_root, policy):
    return {"status": "pass", "base_ref": base_ref, "violations": []}


def test_complete_chain_promotes_atomically(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    result = promote_candidate(
        chain["candidate_run_id"],
        base_ref="HEAD",
        expected_current_sha256=None,
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
        integrity_checker=passing_integrity,
    )
    assert result["status"] == "promoted"
    current = json.loads((chain["outputs"] / "frontier/current.json").read_text())
    assert current["candidate_run_id"] == chain["candidate_run_id"]
    assert (chain["outputs"] / "frontier/history" / f"{result['promotion_id']}.json").is_file()


def test_promotion_requires_positive_score_and_headline_gain(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch, improvement=-0.1)
    create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    with pytest.raises(Exception, match="score|event|promotion"):
        promote_candidate(
            chain["candidate_run_id"],
            base_ref="HEAD",
            expected_current_sha256=None,
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
            integrity_checker=passing_integrity,
        )


def test_integrity_failure_blocks_promotion(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    with pytest.raises(FrontierError, match="integrity"):
        promote_candidate(
            chain["candidate_run_id"],
            base_ref="HEAD",
            expected_current_sha256=None,
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
            integrity_checker=lambda *args: {"status": "fail", "violations": ["README.md"]},
        )


def test_promotion_requires_complete_artifact_chain(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    fake = chain["outputs"] / "evals/fake"
    fake.mkdir()
    (fake / "summary.json").write_text(json.dumps({"score": 99, "status": "succeeded"}))
    with pytest.raises(Exception):
        promote_candidate(
            "fake",
            base_ref="HEAD",
            expected_current_sha256=None,
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
            integrity_checker=passing_integrity,
        )
