---
description: Mine fixed-category statistics from a verified development evaluation
argument-hint: "<dev-eval-run-id>"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli failures mine *)
disable-model-invocation: true
---

Mine only categories and statistics from a verified development evaluation.

```bash
uv run --locked python -m boldt_posttrain.cli failures mine --eval-run "$ARGUMENTS"
```
