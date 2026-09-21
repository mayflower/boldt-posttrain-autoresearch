import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The verification evidence table records commands that were actually executed on
# the original host, including its pre-uv invocation forms. Rewriting it would
# misstate what ran, so it is documentation of the past rather than instruction.
HISTORICAL_DOCS = {"docs/implementation-report.md"}


def operative_docs() -> list[Path]:
    paths = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "AUTORESEARCH_POSTTRAIN.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / ".claude/commands").glob("*.md")),
        ROOT / "scripts/README.md",
    ]
    return [p for p in paths if p.relative_to(ROOT).as_posix() not in HISTORICAL_DOCS]


def test_sync_script_is_uv_only_and_fails_closed():
    script = (ROOT / "scripts/sync_env.sh").read_text(encoding="utf-8")
    assert "uv sync --locked --all-extras" in script
    assert "torch.cuda.is_available()" in script
    # The merge and evaluation levers resolve these from PATH, not from an import.
    assert 'for executable in ("mergekit-yaml", "lm-eval")' in script
    assert not (ROOT / "scripts/sync_conda_env.sh").exists()


def test_documented_cli_invocations_go_through_uv_run():
    bare = re.compile(r"(?<!uv run --locked )python -m boldt_posttrain\.cli")
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in operative_docs()
        if bare.search(path.read_text(encoding="utf-8"))
    ]
    assert violations == []


def test_operative_surface_has_no_conda_dependency():
    # Prose may still mention Conda to explain what was replaced; what must be
    # gone are the operative markers that would make a run depend on it.
    markers = ("conda activate", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "sync_conda_env.sh")
    sources = [
        *operative_docs(),
        *sorted((ROOT / "scripts").glob("*.sh")),
        *sorted((ROOT / "src/boldt_posttrain").rglob("*.py")),
        ROOT / ".claude/settings.json",
        *sorted(p for p in (ROOT / ".claude/hooks").iterdir() if p.is_file()),
    ]
    violations = [
        (path.relative_to(ROOT).as_posix(), marker)
        for path in sources
        for marker in markers
        if marker in path.read_text(encoding="utf-8")
    ]
    assert violations == []


def test_hooks_do_not_depend_on_an_activated_interpreter():
    # NixOS provides python3, never a bare `python`; only an activated venv does.
    # A hook that shells out to `python` would block every tool call there.
    settings = (ROOT / ".claude/settings.json").read_text(encoding="utf-8")
    assert '"command": "python ' not in settings
    assert "python3 " in settings
