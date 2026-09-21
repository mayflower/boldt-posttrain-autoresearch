---
description: Execute one exact deterministic trial
argument-hint: "dry|real <exact-run-id>"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli eval run *) Bash(uv run --locked python -m boldt_posttrain.cli score *) Read
disable-model-invocation: true
---
Evaluate the exact candidate and pass its newly returned eval ID to score. Never reuse an earlier score artifact.

```bash
uv run --locked python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate "$CANDIDATE"
uv run --locked python -m boldt_posttrain.cli score --candidate "$EVAL_RUN_ID"
```
