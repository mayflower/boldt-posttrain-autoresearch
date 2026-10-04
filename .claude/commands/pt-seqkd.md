---
description: Plan or run one sequence-level distillation teacher generation (seqkd stage 1)
argument-hint: "dry|real"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli seqkd generate *) Read Edit(configs/posttrain/secure-current.json) Edit(configs/posttrain/experiments/*.json)
disable-model-invocation: true
---
The teacher must be an exact `policy.teachers` entry. Generation runs outside the loop budget
(about 2-3 hours for 10,000 prompts) and always forwards GPU permission:

```bash
uv run --locked python -m boldt_posttrain.cli seqkd generate --real --allow-gpu --config configs/posttrain/experiments/seqkd-qwen3.8-27b-de.json
```

On success, set `seqkd.generation_run` to the returned run ID in that experiment file and copy
its `experiment` and `seqkd` blocks into `configs/posttrain/secure-current.json`, so `/pt-run`
can run the `seqkd` lever. For a plan use `--dry-run` only. Emit the JSON result unchanged and
stop on a nonzero exit.
