"""Round reports and progress lines are rendered from published artifacts only."""

from boldt_posttrain.loop import experiment_snapshot
from boldt_posttrain.report import progress, render_loop_report
from boldt_posttrain.scoring import create_score
from tests.artifact_chain import complete_chain

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_explains_metrics_gates_and_decision(tmp_path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch, improvement=0.1)
    score = create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    document = json.loads((ROOT / "configs/posttrain/secure-current.json").read_text())
    verdict = {
        "loop_id": "loop-20260101T000000.000000Z-0123456789abcdef",
        "status": "rejected",
        "disposition": "rejected",
        "started_at": "2026-01-01T00:00:00+00:00",
        "duration_seconds": 125.0,
        "budget_seconds": 5400.0,
        "base_ref": "a" * 40,
        "error": None,
        "experiment": experiment_snapshot(document),
        "candidate_run_id": chain["candidate_run_id"],
        "score_run_id": score["score_run_id"],
        "stages": {"integrity": {"status": "pass", "violations": []}},
    }
    report = render_loop_report(verdict, policy=chain["policy"], outputs_root=chain["outputs"])
    assert document["experiment"]["hypothesis"] in report
    assert "learning_rate=1e-05" in report
    assert "| german_instruction | 0.500 | 0.600 | +0.100 |" in report
    assert "pass (Δ +0.100 >= +0.001)" in report
    assert f"Score: **{score['score']:+.3f}**" in report
    assert chain["candidate_run_id"] in report and "Not promoted." in report


def test_report_of_a_failed_round_names_the_error(tmp_path):
    from boldt_posttrain.policy import load_policy

    verdict = {
        "loop_id": "loop-20260101T000000.000000Z-0123456789abcdef",
        "status": "failed",
        "disposition": None,
        "duration_seconds": 3.0,
        "budget_seconds": 5400.0,
        "error": "ScoringError: evaluation policy hash is stale or different",
        "stages": {},
    }
    report = render_loop_report(verdict, policy=load_policy(), outputs_root=tmp_path)
    assert "Status: **failed**" in report and "policy hash is stale" in report


def test_progress_goes_to_stderr_not_stdout(capsys):
    progress("training: step 3/10")
    captured = capsys.readouterr()
    assert captured.out == "" and "training: step 3/10" in captured.err
