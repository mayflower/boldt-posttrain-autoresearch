---
description: Train the verified RLOO lever with mechanical rewards
argument-hint: "[--budget-minutes N]"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli train rlvr *)
disable-model-invocation: true
---

Run the verified RLOO online-RL method with explicit hardware and checkpoint authorization.

```bash
uv run --locked python -m boldt_posttrain.cli train rlvr --real --allow-gpu --allow-checkpoints \
  --config configs/posttrain/secure-current.json --budget-minutes 90 "$ARGUMENTS"
```
