import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands() -> dict[str, str]:
    return {path.name: path.read_text() for path in (ROOT / ".claude/commands").glob("pt-*.md")}


def test_slash_commands_forward_required_real_flags():
    docs = commands()
    assert "--real --allow-gpu" in docs["pt-baseline.md"]
    assert "--real --allow-gpu" in docs["pt-eval.md"]
    assert "--real --allow-gpu --allow-checkpoints" in docs["pt-train.md"]
    assert "data discover --real" in docs["pt-data.md"]


def test_commands_have_no_error_swallowing_or_latest_alias():
    combined = "\n".join(commands().values())
    assert ("||" + " true") not in combined
    assert "--candidate latest" not in combined
    assert not (ROOT / ".claude/commands/pt-bootstrap.md").exists()


def test_autonomous_command_has_narrow_write_surface():
    document = commands()["pt-run.md"]
    assert "Edit(configs/posttrain/secure-current.json)" in document
    assert "Edit(configs/posttrain/experiments/*.json)" in document
    for forbidden in ("Edit(src/", "Write(outputs/", "sed -i", "python -c", "tee "):
        assert forbidden not in document


def test_pretool_guard_denies_write_and_shell_bypass():
    path = ROOT / ".claude/hooks/guard_posttrain.py"
    spec = importlib.util.spec_from_file_location("guard_posttrain", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    assert (
        module.allowed(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/boldt_posttrain/scoring.py"}}
        )[0]
        is False
    )
    assert (
        module.allowed(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "configs/posttrain/secure-current.json"},
            }
        )[0]
        is True
    )
    approved = "uv run --locked python -m boldt_posttrain.cli status"
    assert module.allowed({"tool_name": "Bash", "tool_input": {"command": approved}})[0] is True
    # Redirection stays forbidden even on an otherwise approved command.
    assert (
        module.allowed({"tool_name": "Bash", "tool_input": {"command": approved + " > result"}})[0]
        is False
    )
    # The bare interpreter form is rejected.
    assert (
        module.allowed(
            {"tool_name": "Bash", "tool_input": {"command": "python -m boldt_posttrain.cli status"}}
        )[0]
        is False
    )
    json.loads((ROOT / ".claude/settings.json").read_text())


def test_guard_accepts_documented_absolute_paths_and_notes_but_rejects_escapes():
    path = ROOT / ".claude/hooks/guard_posttrain.py"
    spec = importlib.util.spec_from_file_location("guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for allowed in (
        ROOT / "configs/posttrain/secure-current.json",
        ROOT / "docs/experiments/trial.md",
    ):
        assert module.allowed({"tool_name": "Write", "tool_input": {"file_path": str(allowed)}})[0]
    for denied in (
        "configs/posttrain/policy.json",
        "docs/experiments/../../policy.md",
        "/tmp/trial.json",
    ):
        assert not module.allowed({"tool_name": "Write", "tool_input": {"file_path": denied}})[0]
    assert not module.allowed(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": "uv run --locked python -m boldt_posttrain.cli status\nrm file"
            },
        }
    )[0]


def run_guard_hook(document: dict) -> subprocess.CompletedProcess:
    """Invoke the hook the way Claude Code does: JSON on stdin, protocol on stdout."""
    return subprocess.run(
        [sys.executable, str(ROOT / ".claude/hooks/guard_posttrain.py")],
        input=json.dumps(document),
        capture_output=True,
        text=True,
        check=False,
    )


def test_guard_hook_speaks_the_pretooluse_protocol():
    # A denial must use hookSpecificOutput.permissionDecision; a top-level
    # {"decision": "deny"} is rejected by Claude Code as malformed and the call proceeds.
    denied = run_guard_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "configs/posttrain/policy.json"}}
    )
    assert denied.returncode == 0
    output = json.loads(denied.stdout)
    assert set(output) == {"hookSpecificOutput"}
    assert output["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert output["hookSpecificOutput"]["permissionDecisionReason"]

    shell = run_guard_hook({"tool_name": "Bash", "tool_input": {"command": "git log"}})
    assert json.loads(shell.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"

    # Allowed calls emit nothing, so the regular permission settings still apply.
    for document in (
        {"tool_name": "Read", "tool_input": {"file_path": "AGENTS.md"}},
        {"tool_name": "Write", "tool_input": {"file_path": "docs/experiments/trial.md"}},
        {
            "tool_name": "Bash",
            "tool_input": {"command": "uv run --locked python -m boldt_posttrain.cli status"},
        },
    ):
        allowed = run_guard_hook(document)
        assert allowed.returncode == 0
        assert allowed.stdout.strip() == ""
