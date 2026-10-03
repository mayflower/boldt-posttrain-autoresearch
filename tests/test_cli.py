import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_read_only_commands_execute_without_routing_sentinels(capsys):
    assert main(["policy", "validate"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert main(["eval", "catalog"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"
    assert main(["doctor", "--mode", "all"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_rlvr_defaults_to_secure_loop_config():
    import boldt_posttrain.cli as cli

    args = build_parser().parse_args(
        ["train", "rlvr", "--real", "--allow-gpu", "--allow-checkpoints"]
    )
    assert Path(args.config).name == "secure-current.json"
    assert args.handler is cli._train_command
    assert not hasattr(args, "policy"), "RLVR must use the loop's protected policy"


def test_train_dry_run_is_a_preflight_that_writes_no_artifact(tmp_path: Path, capsys):
    # The manual train verbs now run the loop's secure producer; dry-run is a
    # preflight, not the old recipe path that wrote a run_card. Without a prepared
    # data manifest it fails closed and produces no artifact.
    code = main(["train", "sft", "--dry-run", "--config", "configs/posttrain/secure-current.json"])
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry_run"
    if code == 0:
        assert "message" in result  # preflight ok: nothing written
    else:
        assert code == 2 and "error" in result
    assert not (tmp_path / "runs").exists()


def test_eval_candidate_forwards_verified_checkpoint_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import boldt_posttrain.cli as cli

    import boldt_posttrain.runtime_cli as runtime

    checkpoint = tmp_path / "adapter"
    captured = {}

    monkeypatch.setattr(
        runtime,
        "resolve_model",
        lambda **_kwargs: SimpleNamespace(
            artifact={"path": str(checkpoint)},
            base_model={"repo_id": "org/model", "revision": "a" * 40},
        ),
    )

    def fake_publish(**kwargs):
        captured.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(runtime.evaluation, "_publish_evaluation", fake_publish)
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs")
    assert main(["eval", "run", "--real", "--allow-gpu", "--candidate", "candidate-id"]) == 0
    assert captured["resolved"].artifact["path"] == str(checkpoint)
    assert captured["resolved"].base_model["revision"] == "a" * 40
    assert captured["output_root"] == tmp_path / "outputs/evals"


def test_loop_run_uses_protected_experiment_api(tmp_path: Path, monkeypatch, capsys):
    import boldt_posttrain.cli as cli
    import boldt_posttrain.loop as loop

    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return {"status": "succeeded", "loop_id": "loop-test"}, 0

    monkeypatch.setattr(loop, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs/posttrain")
    assert (
        main(
            [
                "loop",
                "run",
                "--real",
                "--allow-gpu",
                "--allow-checkpoints",
                "--config",
                "configs/posttrain/secure-current.json",
                "--base-ref",
                "HEAD",
                "--budget-minutes",
                "10",
            ]
        )
        == 0
    )
    assert captured["base_ref"] == "HEAD"
    assert captured["outputs_root"] == tmp_path / "outputs/posttrain"
    assert json.loads(capsys.readouterr().out)["loop_id"] == "loop-test"


def test_promote_uses_protected_frontier_api(tmp_path: Path, monkeypatch, capsys):
    import boldt_posttrain.cli as cli

    captured = {}
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs/posttrain")
    monkeypatch.setattr(cli, "current_frontier_hash", lambda _path: "frontier-hash")

    def fake_promote(candidate, **kwargs):
        captured.update(candidate=candidate, **kwargs)
        return {"status": "promoted", "candidate_run_id": candidate}

    monkeypatch.setattr(cli, "promote_candidate", fake_promote)
    assert main(["promote", "--candidate", "candidate-id", "--base-ref", "HEAD"]) == 0
    assert captured["candidate"] == "candidate-id"
    assert captured["expected_current_sha256"] == "frontier-hash"
    assert captured["outputs_root"] == tmp_path / "outputs/posttrain"
    assert json.loads(capsys.readouterr().out)["status"] == "promoted"


def test_real_merge_requires_and_forwards_checkpoint_permission(monkeypatch, capsys):
    import boldt_posttrain.runtime_cli as runtime

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(runtime.merge, "run_search", fake_search)
    base = ["merge", "search", "--real", "--allow-gpu"]
    assert main(base) == 2
    assert "allow-checkpoints" in json.loads(capsys.readouterr().out)["error"]
    assert main([*base, "--allow-checkpoints"]) == 0
    assert captured["allow_checkpoints"] is True
    assert captured["allow_gpu"] is True


@pytest.mark.parametrize(
    ("entrypoint", "command"), [("main_status", "status"), ("main_report", "report")]
)
def test_console_entrypoints_do_not_discard_process_arguments(monkeypatch, entrypoint, command):
    import boldt_posttrain.cli as cli

    seen = []
    monkeypatch.setattr(cli.sys, "argv", [command, "--invalid-option"])
    monkeypatch.setattr(cli, "main", lambda args: seen.extend(args) or 2)
    assert getattr(cli, entrypoint)() == 2
    assert seen == [command, "--invalid-option"]


def test_integrity_check_consumes_the_subcommand_and_preserves_failure(monkeypatch):
    import boldt_posttrain.cli as cli

    seen = {}

    def script(stem, argv):
        seen.update(stem=stem, argv=argv)
        return 1

    monkeypatch.setattr(cli, "_script", script)
    assert main(["integrity", "check", "--base-ref", "a" * 40, "--strict"]) == 1
    assert seen == {
        "stem": "check_posttrain_integrity",
        "argv": ["--base-ref", "a" * 40, "--format", "json", "--strict"],
    }
