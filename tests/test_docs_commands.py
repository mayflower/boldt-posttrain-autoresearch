import shlex
from pathlib import Path

from boldt_posttrain.cli import build_parser
from boldt_posttrain.config import load_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_documented_cli_examples_parse_successfully():
    documents = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "AGENTS.md",
        ROOT / "AUTORESEARCH_POSTTRAIN.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / ".claude/commands").glob("*.md")),
    ]
    commands: list[tuple[Path, str]] = []
    prefix = "uv run --locked python -m boldt_posttrain.cli "
    for document in documents:
        lines = document.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            index += 1
            if not stripped.startswith(prefix):
                continue
            # Join backslash continuations so a wrapped example is parsed whole
            # instead of as a fragment plus a stray trailing backslash.
            while stripped.endswith("\\") and index < len(lines):
                stripped = stripped[:-1] + lines[index].strip()
                index += 1
            commands.append((document, stripped.removeprefix(prefix)))
    assert commands
    parser = build_parser()
    failures = []
    for document, command in commands:
        argv = shlex.split(command)
        # "$ARGUMENTS" is the slash-command placeholder. After an option it is
        # that option's value; standing alone at the end it is "extra flags the
        # caller may append" and has no counterpart in the parser.
        if len(argv) > 1 and argv[-1] == "$ARGUMENTS" and not argv[-2].startswith("--"):
            argv.pop()
        argv = ["ARGUMENTS" if token == "$ARGUMENTS" else token for token in argv]
        try:
            args = parser.parse_args(argv)
            if (
                args.command in {"train", "distill", "loop", "baseline", "merge"}
                or (args.command == "data" and args.action in {"discover", "prepare"})
                or (args.command == "eval" and args.action == "run")
            ):
                if "$" not in args.config and args.config != "ARGUMENTS":
                    load_experiment(ROOT / args.config)
        except Exception as exc:
            failures.append(f"{document.relative_to(ROOT)}: {command}: {exc}")
    assert failures == []
