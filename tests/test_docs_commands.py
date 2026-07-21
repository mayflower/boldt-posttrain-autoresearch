import shlex
from pathlib import Path

from boldt_posttrain.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_documented_cli_examples_parse_successfully():
    documents = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "AUTORESEARCH_POSTTRAIN.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / ".claude/commands").glob("*.md")),
    ]
    commands: list[tuple[Path, str]] = []
    prefix = "python -m boldt_posttrain.cli "
    for document in documents:
        for line in document.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                commands.append((document, stripped.removeprefix(prefix)))
    assert commands
    parser = build_parser()
    failures = []
    for document, command in commands:
        try:
            parser.parse_args(shlex.split(command))
        except Exception as exc:
            failures.append(f"{document.relative_to(ROOT)}: {command}: {exc}")
    assert failures == []
