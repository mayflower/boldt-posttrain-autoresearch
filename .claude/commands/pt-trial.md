---
description: Execute one exact deterministic trial
argument-hint: "dry|real <exact-run-id>"
allowed-tools: Bash(python -m boldt_posttrain.cli eval run *) Bash(python -m boldt_posttrain.cli score *) Read
disable-model-invocation: true
---
Evaluate the exact candidate and pass its newly returned eval ID to score. Never reuse an earlier score artifact.

```bash
python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate "$CANDIDATE"
python -m boldt_posttrain.cli score --candidate "$EVAL_RUN_ID"
```
