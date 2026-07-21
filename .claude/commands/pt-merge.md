---
description: Run or plan verified merge search
argument-hint: "dry|real"
allowed-tools: Bash(python -m boldt_posttrain.cli merge search *) Read
disable-model-invocation: true
---

```bash
python -m boldt_posttrain.cli merge search --real --allow-checkpoints --allow-gpu --config configs/posttrain/current.json --budget-minutes 90
```

For a plan use only `--dry-run`. Emit JSON unchanged.
