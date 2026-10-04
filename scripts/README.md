The supported interface is `uv run --locked python -m boldt_posttrain.cli`; the
`.claude/commands/pt-*.md` wrappers use it directly.

- `build_eval_suite.py` regenerates the protected `german-core-v2` evaluation suite.
- `check_posttrain_integrity.py` fails a change that touches a protected surface; `pt integrity
  check` runs it.
- `sync_env.sh` syncs the locked environment and checks CUDA and the merge/eval executables.

Validate with `uv run --locked --all-extras pytest` and
`uv run --locked --all-extras ruff check src tests scripts`.
