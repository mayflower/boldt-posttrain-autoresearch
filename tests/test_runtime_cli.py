"""Exercise the CLI against actual canonical publishers and verification gates."""

import json
from pathlib import Path

import pytest

from tests.artifact_chain import complete_chain
from boldt_posttrain import cli, evaluation, loop, runtime_cli
from boldt_posttrain.artifacts import EventLog


def test_manual_eval_and_score_share_the_verified_loop_artifact_chain(
    tmp_path, monkeypatch, capsys
):
    chain = complete_chain(tmp_path, monkeypatch)
    config_path = evaluation.ROOT / "configs/posttrain/secure-current.json"
    monkeypatch.setattr(cli, "ROOT", chain["repository"])
    monkeypatch.setattr(cli, "OUTPUTS", chain["outputs"])
    monkeypatch.setattr(runtime_cli, "load_policy", lambda: chain["policy"])
    monkeypatch.setattr(loop, "load_policy", lambda: chain["policy"])

    chain_records = evaluation.generate_cases

    def records(resolved, cases, *, device, deadline):
        assert deadline > 0
        return chain_records(resolved, cases, device=device)

    monkeypatch.setattr(evaluation, "generate_cases", records)
    # Only model execution is stubbed; resolution, publishing and scoring are real.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert (
        cli.main(
            [
                "eval",
                "run",
                "--real",
                "--allow-gpu",
                "--candidate",
                chain["candidate_run_id"],
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["run_id"].startswith("eval-")
    assert cli.main(["score", "--candidate", evaluated["run_id"]]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["status"] == "passed"
    assert scored["score_run_id"].startswith("score-")
    EventLog(chain["outputs"]).validate()
    assert cli.main(["status"]) == 0
    capsys.readouterr()

    summary_path = chain["outputs"] / "evals" / evaluated["run_id"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    raw = Path(summary["raw_generations"]["path"])
    if not raw.is_absolute():
        raw = chain["repository"] / raw
    raw.write_text("tampered\n")
    assert cli.main(["score", "--candidate", evaluated["run_id"]]) == 5
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize("verb", [["eval", "run", "--model", "seed"], ["merge", "search"]])
@pytest.mark.parametrize("budget", ["nan", "inf", "0", "-1"])
def test_manual_evaluation_and_merge_reject_invalid_deadlines(verb, budget, capsys):
    assert cli.main([*verb, "--dry-run", "--budget-minutes", budget]) == 2
    assert "positive and finite" in json.loads(capsys.readouterr().out)["error"]


def test_data_adapter_preserves_producer_failure_and_output_namespace(
    tmp_path, monkeypatch, capsys
):
    captured = {}

    def fail(args, **kwargs):
        captured.update(action=args.data_command, **kwargs)
        return {"status": "failed", "error": "stream interrupted"}, 4

    monkeypatch.setattr(runtime_cli.data_pipeline, "run_cli", fail)
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs")
    assert cli.main(["data", "prepare", "--real"]) == 4
    assert captured == {"action": "prepare", "outputs_root": tmp_path / "outputs"}
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
