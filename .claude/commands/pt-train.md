---
description: Run or plan one SFT, CPT, or preference training job
argument-hint: "dry|real sft|cpt|preference"
allowed-tools: Bash(python -m boldt_posttrain.cli train *) Read Edit(configs/posttrain/current.json)
disable-model-invocation: true
---
Real training always forwards all permissions:

```bash
python -m boldt_posttrain.cli train sft --real --allow-gpu --allow-checkpoints --config configs/posttrain/current.json --budget-minutes 90
```

Select `cpt` or `preference --method dpo|kto|orpo` explicitly. For a plan use `--dry-run` only.
