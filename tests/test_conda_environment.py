from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_conda_sync_preserves_existing_cuda_torch():
    script = (ROOT / "scripts/sync_conda_env.sh").read_text(encoding="utf-8")
    assert 'CONDA_DEFAULT_ENV:-}" != "boldtembed"' in script
    assert 'before" != "2.6.0+cu124 12.4"' in script
    assert "uv sync --active --all-extras --inexact --no-install-package torch --locked" in script
    assert 'after" != "$before"' in script


def test_local_documentation_uses_active_conda_python_directly():
    paths = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "AUTORESEARCH_POSTTRAIN.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / ".claude/commands").glob("*.md")),
    ]
    forbidden = "uv run " + "python -m boldt_posttrain.cli"
    violations = [
        path.relative_to(ROOT).as_posix() for path in paths if forbidden in path.read_text()
    ]
    assert violations == []
