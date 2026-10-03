---
description: Run the default-deny integrity gate
argument-hint: "--base-ref REF"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli integrity check *) Bash(uv run --locked git status --short) Bash(uv run --locked git diff --name-only *) Read
disable-model-invocation: true
---

```bash
uv run --locked python -m boldt_posttrain.cli integrity check --base-ref "$BASE_REF"
```

A nonzero exit is the verdict and must be preserved.
