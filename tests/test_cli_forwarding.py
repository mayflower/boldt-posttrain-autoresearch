"""The CLI reaches its target scripts through a second argparse pass.

tests/test_cli.py replaces cli._script with a stub, and tests/test_docs_commands.py
stops at build_parser(). Neither crosses the boundary where the forwarded argv is
parsed again, which is where a type mismatch turns a documented command into exit 2.
"""

import pytest

from boldt_posttrain import cli, training


TRAIN_ACTIONS = [
    pytest.param("sft", [], id="sft"),
    pytest.param("cpt", [], id="cpt"),
    pytest.param("preference", ["--method", "dpo"], id="preference"),
]


@pytest.mark.parametrize(("action", "extra"), TRAIN_ACTIONS)
def test_documented_budget_survives_forwarding_to_the_trainer(monkeypatch, action, extra):
    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(training, "run_training_trial", record)
    exit_code = cli.main(
        [
            "train",
            action,
            "--dry-run",
            "--config",
            "configs/posttrain/current.json",
            "--budget-minutes",
            "90",
            *extra,
        ]
    )
    assert exit_code == 0
    assert float(seen["budget_minutes"]) == 90.0


@pytest.mark.parametrize(("action", "extra"), TRAIN_ACTIONS)
def test_fractional_budget_is_forwarded_without_truncation(monkeypatch, action, extra):
    seen: dict = {}
    monkeypatch.setattr(training, "run_training_trial", lambda **kw: seen.update(kw) or 0)
    assert (
        cli.main(
            [
                "train",
                action,
                "--dry-run",
                "--config",
                "configs/posttrain/current.json",
                "--budget-minutes",
                "7.5",
                *extra,
            ]
        )
        == 0
    )
    assert float(seen["budget_minutes"]) == 7.5
