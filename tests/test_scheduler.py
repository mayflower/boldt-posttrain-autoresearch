import pytest

from boldt_posttrain.data_pipeline import IntegrityError
from boldt_posttrain.scheduler import run_successive_halving


def test_six_trials_reduce_to_two_then_one_serially():
    plan = {
        "trials": [{"trial_id": f"t{i}", "overrides": {"learning_rate": 1e-5}} for i in range(6)]
    }
    calls = []

    def train(trial, rung, parent):
        calls.append((rung.index, trial["trial_id"], None if parent is None else parent["run_id"]))
        return {
            "status": "ok",
            "technical_error_count": 0,
            "hard_gates_passed": True,
            "proxy_score": 10 - int(trial["trial_id"][1:]),
            "gpu_seconds": 1,
            "run_id": f"{trial['trial_id']}-r{rung.index}",
        }

    result = run_successive_halving(
        plan=plan,
        full_budget=1000,
        train_and_proxy=train,
        dev_evaluate=lambda _winner: {"status": "ok", "technical_error_count": 0},
    )
    assert result["status"] == "ok"
    assert [sum(rung == i for rung, *_ in calls) for i in (1, 2, 3)] == [6, 2, 1]
    assert result["runs"][-1]["continuation_mode"] == "adapter_fresh_optimizer"


def test_integrity_error_stops_search_immediately():
    plan = {"trials": [{"trial_id": "a", "overrides": {"learning_rate": 1e-5}}]}

    def corrupt(_trial, _rung, _parent):
        raise IntegrityError("tampered manifest")

    with pytest.raises(IntegrityError):
        run_successive_halving(
            plan=plan,
            full_budget=1000,
            train_and_proxy=corrupt,
            dev_evaluate=lambda _winner: {},
        )


def test_successful_but_ungated_trials_are_rejected_not_technical_failure():
    plan = {"trials": [{"trial_id": "a", "overrides": {"learning_rate": 1e-5}}]}
    result = run_successive_halving(
        plan=plan,
        full_budget=1000,
        train_and_proxy=lambda _trial, _rung, _parent: {
            "status": "ok",
            "technical_error_count": 0,
            "hard_gates_passed": False,
            "proxy_score": 1.0,
            "gpu_seconds": 1.0,
        },
        dev_evaluate=lambda _winner: {},
    )
    assert result["status"] == "rejected"
