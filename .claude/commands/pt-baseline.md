---
description: Run or plan the immutable seed baseline
argument-hint: "dry|real"
allowed-tools: Bash(python -m boldt_posttrain.cli baseline run *) Read
disable-model-invocation: true
---
Invoke exactly one mode. Real execution:

```bash
python -m boldt_posttrain.cli baseline run --real --allow-gpu --config configs/posttrain/current.json
```

For a plan replace the execution flags with `--dry-run`.
