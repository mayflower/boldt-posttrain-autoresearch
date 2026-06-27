---
description: Train one LoRA/QLoRA specialist branch from the Boldt warm start
argument-hint: "[dry|real] [general-de|reasoning-de|coding-de|safety-de|longcontext-de|raw-quality-de]"
allowed-tools: Bash(python scripts/pt_train_specialist.py *) Bash(python scripts/pt_train_cpt.py *) Bash(python scripts/pt_train_preference.py *) Bash(python scripts/pt_eval.py *) Bash(python scripts/check_posttrain_integrity.py *) Bash(cat *) Read Edit
disable-model-invocation: true
---
# PostTrain AutoResearch — train specialist

Parse `$ARGUMENTS`: mode default `dry`; specialist default from `configs/posttrain/current.json`. Set `MODE_FLAG=--dry-run` or `MODE_FLAG=--real` before running commands.

Allowed specialists:

- `general-de`
- `reasoning-de`
- `coding-de`
- `safety-de`
- `longcontext-de`
- `raw-quality-de`
- `preference-de`

Before real training, verify data manifest exists and is clean:

```bash
cat outputs/posttrain/data/manifest.json
cat outputs/posttrain/data/leakage_report.json
```

Run the corresponding training script:

```bash
python scripts/pt_train_specialist.py \
  --config configs/posttrain/current.json \
  --specialist "$SPECIALIST" \
  --out outputs/posttrain/runs \
  --budget-minutes 90 \
  "$MODE_FLAG"
```

For `raw-quality-de`, use `pt_train_cpt.py`. For `preference-de`, use `pt_train_preference.py`.

If a candidate checkpoint/adaptor is produced, evaluate it:

```bash
python scripts/pt_eval.py --config configs/posttrain/current.json --candidate latest --out outputs/posttrain/evals "$MODE_FLAG" || true
```

Run integrity and report artifacts. Never call a dry-run metric a quality result.
