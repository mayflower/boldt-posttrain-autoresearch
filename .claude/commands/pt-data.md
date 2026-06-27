---
description: Discover and prepare German parts of openeurollm datasets with manifest/license/leakage checks
argument-hint: "[dry|real]"
allowed-tools: Bash(python scripts/pt_discover_openeurollm_de.py *) Bash(python scripts/pt_prepare_openeurollm_de.py *) Bash(cat *) Bash(tail *) Bash(ls *) Read Edit
disable-model-invocation: true
---
# PostTrain AutoResearch — data

Parse `$ARGUMENTS`: mode is `dry` by default; `real` only if explicitly present.

1. Discover German OpenEuroLLM candidates without full downloads:

```bash
python scripts/pt_discover_openeurollm_de.py \
  --config configs/posttrain/current.json \
  --out outputs/posttrain/data/discovery.json \
  --format markdown \
  --dry-run
```

2. If mode is `real`, prepare trainable shards and require license + leakage checks:

```bash
python scripts/pt_prepare_openeurollm_de.py \
  --config configs/posttrain/current.json \
  --discovery outputs/posttrain/data/discovery.json \
  --out outputs/posttrain/data \
  --real
```

If mode is dry, run the prepare script in dry mode:

```bash
python scripts/pt_prepare_openeurollm_de.py \
  --config configs/posttrain/current.json \
  --discovery outputs/posttrain/data/discovery.json \
  --out outputs/posttrain/data \
  --dry-run
```

Report: candidate datasets/configs/splits, schema guesses, German-filter method, license status, leakage status, and whether the data is trainable.

Do not train in this command.
