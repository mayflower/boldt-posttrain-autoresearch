"""The manual `train` verbs run the loop's single producer in-process.

They used to forward parsed args back into a second argparse pass in
scripts/pt_train_*.py, which is where the float/int `--budget-minutes` mismatch
lived and where the recipe run-card (unresolvable by the resolver) was written.
Both are gone: `train` now calls `loop.train_one_lever`, the same producer the
loop uses, so a manual candidate is resolver-compatible. These tests pin the
wiring without needing a GPU.
"""

import json

import pytest

from boldt_posttrain import cli, loop


TRAIN_ACTIONS = [
    pytest.param("sft", [], id="sft"),
    pytest.param("cpt", [], id="cpt"),
    pytest.param("preference", ["--method", "dpo"], id="preference"),
]


@pytest.mark.parametrize(("action", "extra"), TRAIN_ACTIONS)
def test_real_train_calls_the_single_producer_with_the_named_lever(monkeypatch, action, extra):
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"status": "succeeded", "run_id": "x"}, 0

    monkeypatch.setattr(loop, "train_one_lever", fake)
    code = cli.main(
        [
            "train",
            action,
            "--real",
            "--allow-gpu",
            "--allow-checkpoints",
            "--budget-minutes",
            "90",
            *extra,
        ]
    )
    assert code == 0
    assert seen["lever"] == action
    # Float, not int: the old int forwarding rejected the documented "90".
    assert isinstance(seen["budget_minutes"], float) and seen["budget_minutes"] == 90.0


def test_fractional_budget_reaches_the_producer(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        loop, "train_one_lever", lambda **kw: (seen.update(kw) or {"status": "s", "run_id": "x"}, 0)
    )
    assert (
        cli.main(
            [
                "train",
                "sft",
                "--real",
                "--allow-gpu",
                "--allow-checkpoints",
                "--budget-minutes",
                "7.5",
            ]
        )
        == 0
    )
    assert seen["budget_minutes"] == 7.5


def test_dry_run_is_a_preflight_and_never_trains(monkeypatch, capsys):
    called = {"trained": False}

    def fake(**kwargs):
        called["trained"] = True
        return {"status": "succeeded"}, 0

    monkeypatch.setattr(loop, "train_one_lever", fake)
    # No prepared manifest in the test tree -> preflight fails closed with exit 2,
    # and crucially the producer is never invoked.
    code = cli.main(["train", "sft", "--dry-run"])
    assert code == 2
    assert called["trained"] is False
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["mode"] == "dry_run"
