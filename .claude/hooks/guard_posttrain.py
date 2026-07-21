#!/usr/bin/env python3
"""PreToolUse guard for the autonomous post-training trust boundary."""

from __future__ import annotations

import json
import re
import sys


def allowed(document: dict) -> tuple[bool, str]:
    tool = document.get("tool_name", "")
    tool_input = document.get("tool_input", {})
    if tool in {"Read", "Glob", "Grep"}:
        return True, "read-only tool"
    if tool in {"Edit", "Write"}:
        path = str(tool_input.get("file_path") or tool_input.get("path") or "").replace("\\", "/")
        editable = path == "configs/posttrain/current.json" or re.fullmatch(
            r"configs/posttrain/experiments/[^/]+\.json", path
        )
        return (
            bool(editable),
            "editable experiment surface"
            if editable
            else "writes are limited to strict experiment files",
        )
    if tool == "Bash":
        command = str(tool_input.get("command", ""))
        if any(token in command for token in (">", "<", "|", ";", "`", "$(")):
            return False, "shell composition and redirection are forbidden"
        allowed_commands = (
            "python -m boldt_posttrain.cli ",
            "git rev-parse HEAD",
            "git status --short",
            "git diff -- ",
        )
        approved = command.startswith(allowed_commands)
        return approved, "approved command" if approved else "command is outside the allowlist"
    return False, "tool is outside the autonomous trust boundary"


def main() -> int:
    document = json.load(sys.stdin)
    is_allowed, reason = allowed(document)
    print(json.dumps({"decision": "allow" if is_allowed else "deny", "reason": reason}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
