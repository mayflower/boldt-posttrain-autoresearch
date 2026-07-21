import hashlib
import json
from pathlib import Path

import pytest

from boldt_posttrain.cli import build_parser, main


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and "plans" not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    "arguments",
    [
        ["policy", "validate"],
        ["integrity", "check", "--base-ref", "HEAD"],
        ["model", "resolve", "--model", "mayflowergmbh/boldt-dc-1b-german-it-16k-dpo"],
        ["data", "discover", "--dry-run"],
        ["baseline", "run", "--real", "--allow-gpu"],
        ["train", "sft", "--real", "--allow-gpu", "--allow-checkpoints"],
        ["eval", "run", "--real", "--allow-gpu", "--candidate", "candidate-id"],
        ["merge", "search", "--real", "--allow-gpu", "--allow-checkpoints"],
        ["promote", "--candidate", "candidate-id", "--base-ref", "HEAD"],
        [
            "loop",
            "run",
            "--real",
            "--allow-gpu",
            "--allow-checkpoints",
            "--base-ref",
            "HEAD",
            "--budget-minutes",
            "10",
        ],
    ],
)
def test_cli_examples_parse(arguments: list[str]):
    build_parser().parse_args(arguments)


def test_mutating_mode_is_never_implicit():
    with pytest.raises(Exception):
        build_parser().parse_args(["data", "discover"])


def test_invalid_experiment_uses_configuration_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import boldt_posttrain.cli as cli

    experiment = tmp_path / "invalid.json"
    experiment.write_text('{"promotion": {}}')
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert main(["data", "discover", "--dry-run", "--config", "invalid.json"]) == 2
    assert json.loads(capsys.readouterr().out)["exit_code"] == 2


def test_dry_run_cannot_touch_real_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import boldt_posttrain.cli as cli

    outputs = tmp_path / "outputs/posttrain"
    real_file = outputs / "baseline/current.json"
    real_file.parent.mkdir(parents=True)
    real_file.write_text("immutable")
    before = tree_hash(outputs)
    monkeypatch.setattr(cli, "OUTPUTS", outputs)
    assert main(["data", "discover", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry_run"
    assert tree_hash(outputs) == before
    assert list((outputs / "plans").glob("*/plan.json"))


def test_unknown_real_candidate_creates_no_eval_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import boldt_posttrain.cli as cli
    import boldt_posttrain.resolver as resolver

    outputs = tmp_path / "outputs/posttrain"
    monkeypatch.setattr(cli, "OUTPUTS", outputs)
    monkeypatch.setattr(resolver, "OUTPUTS", outputs)
    exit_code = main(["eval", "run", "--real", "--allow-gpu", "--candidate", "DOES-NOT-EXIST"])
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert result["status"] == "failed"
    assert not (outputs / "evals").exists()
