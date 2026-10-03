---
description: Plan or run one sequence-level distillation teacher generation (seqkd stage 1)
argument-hint: "dry|real"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli seqkd generate *) Read Edit(configs/posttrain/experiments/*.json)
disable-model-invocation: true
---
The teacher must be an exact `policy.teachers` entry. Generation runs outside the loop
budget and always forwards GPU permission:

```bash
uv run --locked python -m boldt_posttrain.cli seqkd generate --real --allow-gpu --config configs/posttrain/experiments/seqkd-qwen3.8-27b-de.json
```

Pin the returned run ID as `seqkd.generation_run` in the experiment file, then train with
`train seqkd --real --allow-gpu --allow-checkpoints` or run the loop with `lever: seqkd`.
For a plan use `--dry-run` only.
