---
description: Establish or inspect baseline eval for the seed model
argument-hint: "[dry|real]"
allowed-tools: Bash(python scripts/pt_baseline.py *) Bash(python scripts/pt_eval.py *) Bash(cat *) Bash(ls *) Read
disable-model-invocation: true
---
# PostTrain AutoResearch — baseline

Parse `$ARGUMENTS`: mode is `dry` by default; `real` only if explicitly present. Use `MODE_FLAG=--dry-run` for dry mode and `MODE_FLAG=--real` for real mode when running shell examples.

Seed model is read from `configs/posttrain/current.json`.

Run:

```bash
python scripts/pt_baseline.py \
  --config configs/posttrain/current.json \
  --out outputs/posttrain/baseline \
  "$MODE_FLAG"
```

If `pt_baseline.py` is not implemented but `pt_eval.py` is, evaluate the seed model directly:

```bash
python scripts/pt_eval.py \
  --config configs/posttrain/current.json \
  --model mayflowergmbh/boldt-dc-1b-german-it-16k-dpo \
  --label baseline-seed \
  --out outputs/posttrain/baseline \
  "$MODE_FLAG"
```

Report the artifact paths and explicitly mark dry-run outputs as plumbing only.
