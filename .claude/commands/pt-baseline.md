---
description: Create or rebuild the seed baseline every candidate is scored against
argument-hint: "dry|real [replace]"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli baseline run *) Bash(uv run --locked python -m boldt_posttrain.cli status) Read
disable-model-invocation: true
---
Invoke exactly one mode. Real execution:

```bash
uv run --locked python -m boldt_posttrain.cli baseline run --real --allow-gpu --config configs/posttrain/secure-current.json
```

This creates the first baseline and rebuilds one that no longer verifies under the current
policy, suite or stack (the new run card records which baseline it supersedes and why). A
baseline that still verifies is replaced only when the user passes `replace`; then append
`--replace-baseline`. For a plan replace the execution flags with `--dry-run`. Emit the JSON
result unchanged and stop on a nonzero exit.
