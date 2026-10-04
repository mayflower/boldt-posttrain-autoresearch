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
