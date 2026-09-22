# Boldt Post-Training — Agent Contract (Claude Code)

The full, tool-neutral contract lives in `AGENTS.md` and applies here verbatim.
It is imported below so Claude Code loads it automatically:

@AGENTS.md

## Claude Code specifics

- `/pt-*` slash commands in `.claude/commands/` are thin wrappers over the
  `uv run --locked …` commands documented in `AGENTS.md`.
- `.claude/hooks/` enforce the editable surface (the PreToolUse guard) and print a
  short orientation at session start.

These are conveniences; the rules in `AGENTS.md` are authoritative for every agent.
