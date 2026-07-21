import copy
import json
import time
from pathlib import Path

import pytest

from boldt_posttrain import config as config_module
from boldt_posttrain.artifacts import EventLog, new_run_id, sha256_file
from boldt_posttrain.loop import LoopError, _remaining, run_experiment
from tests.artifact_chain import fixture_policy, initialized_repository


def loop_config(tmp_path: Path) -> config_module.ExperimentConfig:
    document = copy.deepcopy(config_module.load_experiment().document)
    document["experiment"]["lever"] = "sft"
    return config_module.ExperimentConfig(tmp_path / "experiment.json", document)


def patch_successful_stages(monkeypatch, tmp_path: Path, *, score_status: str, score_run_id: str):
    import boldt_posttrain.loop as loop

    policy = fixture_policy(tmp_path)
    configured = loop_config(tmp_path)
    candidate_run_id = new_run_id("train-sft")
    eval_run_id = new_run_id("eval")
    monkeypatch.setattr(loop, "load_policy", lambda: policy)
    monkeypatch.setattr(loop.config_module, "load_experiment", lambda path: configured)
    monkeypatch.setattr(loop, "load_baseline", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        loop,
        "verify_data_manifest",
        lambda *args, **kwargs: {
            "status": "trainable",
            "shards": [],
            "license_status": "usable",
            "leakage_statistics": {"status": "clean", "hit_count": 0},
        },
    )
    monkeypatch.setattr(
        loop,
        "_execute_lever",
        lambda *args, **kwargs: {"status": "succeeded", "run_id": candidate_run_id},
    )
    monkeypatch.setattr(loop, "resolve_model", lambda **kwargs: object())
    monkeypatch.setattr(
        loop,
        "_publish_evaluation",
        lambda **kwargs: {"status": "succeeded", "run_id": eval_run_id},
    )
    monkeypatch.setattr(
        loop,
        "create_score",
        lambda *args, **kwargs: {
            "status": score_status,
            "score_run_id": score_run_id,
            "candidate_run_id": candidate_run_id,
            "score": 0.1 if score_status == "passed" else -0.1,
        },
    )
    monkeypatch.setattr(
        loop,
        "_integrity_check",
        lambda *args, **kwargs: {"status": "pass", "violations": []},
    )
    return candidate_run_id


def test_loop_never_reuses_previous_score_artifact(tmp_path: Path, monkeypatch):
    repository = initialized_repository(tmp_path / "repo")
    outputs = repository / "outputs/posttrain"
    stale_score = new_run_id("score")
    (outputs / "scores" / stale_score).mkdir(parents=True)
    (outputs / "scores" / stale_score / "score.json").write_text("{}")
    patch_successful_stages(
        monkeypatch,
        tmp_path,
        score_status="passed",
        score_run_id=stale_score,
    )
    verdict, exit_code = run_experiment(
        config_path=tmp_path / "experiment.json",
        base_ref="HEAD",
        budget_minutes=1,
        promote=False,
        allow_checkpoints=True,
        allow_gpu=True,
        outputs_root=outputs,
        repository_root=repository,
    )
    assert exit_code == 4
    assert verdict["status"] == "failed"
    assert "reuse" in verdict["error"]


def test_disposition_is_set_from_fresh_score_result(tmp_path: Path, monkeypatch):
    repository = initialized_repository(tmp_path / "repo")
    outputs = repository / "outputs/posttrain"
    patch_successful_stages(
        monkeypatch,
        tmp_path,
        score_status="rejected",
        score_run_id=new_run_id("score"),
    )
    verdict, exit_code = run_experiment(
        config_path=tmp_path / "experiment.json",
        base_ref="HEAD",
        budget_minutes=1,
        promote=False,
        allow_checkpoints=True,
        allow_gpu=True,
        outputs_root=outputs,
        repository_root=repository,
    )
    assert exit_code == 1
    assert verdict["status"] == "rejected"
    assert verdict["disposition"] == "rejected"
    path = outputs / "loops" / verdict["loop_id"] / "verdict.json"
    finish = json.loads((outputs / "events.jsonl").read_text().splitlines()[-1])
    assert finish["payload"]["verdict_sha256"] == sha256_file(path)
    EventLog(outputs).validate()


def test_global_deadline_blocks_new_stage():
    with pytest.raises(LoopError, match="budget exhausted"):
        _remaining(time.monotonic() - 1, stage="evaluation")
