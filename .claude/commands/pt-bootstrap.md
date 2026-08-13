---
description: Bootstrap verified discovery, data, and development baseline artifacts
argument-hint: "[--device cuda:N] [--budget-minutes N]"
allowed-tools: Bash(uv run pt bootstrap run *)
disable-model-invocation: true
---
# PostTrain AutoResearch — bootstrap

Run the real artifact-derived bootstrap. It does not train or rewrite the current experiment.

```bash
uv run pt bootstrap run --real --allow-gpu \
  --config configs/posttrain/current.json "$ARGUMENTS"
```

Report the final derived state and the discovery, selection, manifest, decontamination, and
development-baseline artifact paths.
