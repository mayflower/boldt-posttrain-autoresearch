---
description: Train the verified RLOO lever with mechanical rewards
argument-hint: "[--device cuda:N] [--budget-minutes N]"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli train rlvr *)
disable-model-invocation: true
---

Run the single supported online-RL method with explicit hardware and checkpoint authorization.

```bash
uv run --locked python -m boldt_posttrain.cli train rlvr --real --allow-gpu --allow-checkpoints \
  --config configs/posttrain/current.json --budget-minutes 90 "$ARGUMENTS"
```
