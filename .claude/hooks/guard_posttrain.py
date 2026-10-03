#!/usr/bin/env python3
"""PreToolUse guard for the autonomous post-training trust boundary."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def allowed(document: dict) -> tuple[bool, str]:
    tool = document.get("tool_name", "")
    tool_input = document.get("tool_input", {})
    if tool in {"Read", "Glob", "Grep"}:
        return True, "read-only tool"
    if tool in {"Edit", "Write"}:
        requested = str(tool_input.get("file_path") or tool_input.get("path") or "").replace(
            "\\", "/"
        )
        try:
            path = (ROOT / requested).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return False, "write is outside the repository"
        editable = (
            path == "configs/posttrain/secure-current.json"
            or re.fullmatch(r"configs/posttrain/experiments/[^/]+\.json", path)
            or re.fullmatch(r"docs/experiments/[^/]+\.md", path)
        )
        return (
            bool(editable),
            "editable experiment surface"
            if editable
            else "writes are limited to strict experiment files and experiment notes",
        )
    if tool == "Bash":
        command = str(tool_input.get("command", ""))
        if any(token in command for token in (">", "<", "|", ";", "`", "$(", "&", "\n", "\r")):
            return False, "shell composition and redirection are forbidden"
        allowed_commands = (
            # uv owns the environment; the bare interpreter form is not allowed
            # because it depends on a previously activated shell and drops
            # .venv/bin from PATH, which the merge and eval levers need.
            "uv run --locked python -m boldt_posttrain.cli ",
            "uv run --locked git rev-parse HEAD",
            "uv run --locked git status --short",
            "uv run --locked git diff -- ",
        )
        approved = command.startswith(allowed_commands)
        return approved, "approved command" if approved else "command is outside the allowlist"
    return False, "tool is outside the autonomous trust boundary"


def hook_output(is_allowed: bool, reason: str) -> dict | None:
    """Claude Code PreToolUse protocol: deny via hookSpecificOutput.permissionDecision.

    Allowed calls emit nothing, so the regular permission settings still apply. A
    top-level ``{"decision": "deny"}`` is not a valid PreToolUse answer: Claude Code
    rejects it as malformed and lets the tool call through.
    """
    if is_allowed:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def main() -> int:
    document = json.load(sys.stdin)
    output = hook_output(*allowed(document))
    if output is not None:
        print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
