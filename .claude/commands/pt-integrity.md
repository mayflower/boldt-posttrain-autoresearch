---
description: Run the default-deny integrity gate
argument-hint: "--base-ref REF"
allowed-tools: Bash(python -m boldt_posttrain.cli integrity check *) Bash(git status --short) Bash(git diff --name-only *) Read
disable-model-invocation: true
---

```bash
python -m boldt_posttrain.cli integrity check --base-ref "$BASE_REF"
```

A nonzero exit is the verdict and must be preserved.
