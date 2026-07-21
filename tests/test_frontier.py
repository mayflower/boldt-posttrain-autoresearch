import json
from pathlib import Path

import pytest

from boldt_posttrain.frontier import (
    FrontierError,
    frontier_status,
    promote_candidate,
    verify_frontier,
)
from boldt_posttrain.scoring import create_score
from tests.artifact_chain import complete_chain
from tests.test_promotion import passing_integrity


def promoted_chain(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    promotion = promote_candidate(
        chain["candidate_run_id"],
        base_ref="HEAD",
        expected_current_sha256=None,
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
        integrity_checker=passing_integrity,
    )
    return chain, promotion


def test_frontier_contains_only_verified_scored_real_candidates(tmp_path: Path, monkeypatch):
    chain, promotion = promoted_chain(tmp_path, monkeypatch)
    status = frontier_status(
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    assert status["verified_promotions"] == 1
    assert status["frontier"]["promotion_id"] == promotion["promotion_id"]
    (chain["outputs"] / "frontier/history/fake.json").write_text(
        json.dumps(
            {
                "candidate_run_id": chain["candidate_run_id"],
                "score_run_id": status["frontier"]["score_run_id"],
            }
        )
    )
    status = frontier_status(
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    assert status["verified_promotions"] == 1


def test_frontier_detects_history_manipulation(tmp_path: Path, monkeypatch):
    chain, promotion = promoted_chain(tmp_path, monkeypatch)
    history = chain["outputs"] / "frontier/history" / f"{promotion['promotion_id']}.json"
    history.write_text("{}")
    with pytest.raises(FrontierError, match="hash"):
        verify_frontier(
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
        )


def test_stale_frontier_hash_rejects_second_promotion(tmp_path: Path, monkeypatch):
    chain, _ = promoted_chain(tmp_path, monkeypatch)
    with pytest.raises(FrontierError, match="stale"):
        promote_candidate(
            chain["candidate_run_id"],
            base_ref="HEAD",
            expected_current_sha256=None,
            policy=chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
            integrity_checker=passing_integrity,
        )
