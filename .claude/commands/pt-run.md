---
description: Agentic outer loop over one deterministic real experiment at a time
argument-hint: "<rounds> real"
allowed-tools: Bash(python -m boldt_posttrain.cli *) Bash(git rev-parse HEAD) Bash(git status --short) Bash(git diff -- *) Read Edit(configs/posttrain/current.json) Edit(configs/posttrain/experiments/*.json) Write(configs/posttrain/experiments/*.json) Glob(configs/posttrain/experiments/*.json) Grep
disable-model-invocation: true
---
Capture `BASE_REF` once. For each serial round, edit only strict experiment files, choose exactly one lever, then invoke:

```bash
python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/current.json --base-ref "$BASE_REF" --budget-minutes 90
```

Never use redirections or general-purpose file mutation commands. Stop on integrity or technical failure and after two non-improving rounds.
