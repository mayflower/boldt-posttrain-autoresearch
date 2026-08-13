---
description: Establish an immutable real development or promotion baseline
argument-hint: "dev|promotion [--device cuda:N] [--budget-minutes N]"
allowed-tools: Bash(uv run pt baseline run *)
disable-model-invocation: true
---
# PostTrain AutoResearch — baseline

Use `dev` unless `promotion` is explicitly requested. Promotion suite contents are read only by
the baseline CLI through the registered hash and `BOLDT_PROMOTION_SUITE`.

```bash
uv run pt baseline run --real --allow-gpu \
  --config configs/posttrain/current.json --profile "$PROFILE" "$ARGUMENTS"
```

The command never replaces an existing immutable baseline pointer.
