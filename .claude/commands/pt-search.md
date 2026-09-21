---
description: Run the serial three-rung Successive Halving search
argument-hint: "<search-config.json> [--device cuda:N]"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli search run *)
disable-model-invocation: true
---

Run the protected serial search plan; do not alter policy or execute trials concurrently.

```bash
uv run --locked python -m boldt_posttrain.cli search run --real --allow-gpu --allow-checkpoints --config "$ARGUMENTS"
```
