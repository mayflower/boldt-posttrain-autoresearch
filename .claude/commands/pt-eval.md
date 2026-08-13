---
description: Run a proxy, development, or promotion evaluation and protected scoring
argument-hint: "proxy|dev|promotion <model> <label> [--device cuda:N]"
allowed-tools: Bash(uv run pt eval run *) Bash(uv run pt score *)
disable-model-invocation: true
---
# PostTrain AutoResearch — evaluation

Run the selected profile through the CLI. Do not read an external promotion suite directly; the
CLI verifies its human-registered hash and keeps raw generations in the protected output area.

```bash
uv run pt eval run --real --allow-gpu --profile "$PROFILE" \
  --model "$MODEL" --label "$LABEL" --config configs/posttrain/current.json
```

Score `dev` against the development baseline for search evidence, or `promotion` against the
promotion baseline for promotion evidence. Proxy is never scored for promotion.
