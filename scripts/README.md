The supported interface is `uv run --locked python -m boldt_posttrain.cli`.
The `.claude/commands/pt-*.md` wrappers use that interface directly.

The `pt_*.py` compatibility entrypoints for data, baseline, eval, score, promote,
merge, loop, distillation and status/report forward to the canonical CLI without
changing exit codes. `pt_log_result.py` is a historical TSV utility, outside the
verified event chain. Other scripts build protected evaluation fixtures, check
integrity, sync the locked environment or run explicit measurements.

Validate with `uv run --locked --all-extras pytest` and
`uv run --locked --all-extras ruff check src tests scripts`.
