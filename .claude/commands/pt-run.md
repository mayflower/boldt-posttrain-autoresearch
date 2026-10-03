---
description: Agentic outer loop over one deterministic real experiment at a time
argument-hint: "<rounds> real"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli *) Bash(uv run --locked git rev-parse HEAD) Bash(uv run --locked git status --short) Bash(uv run --locked git diff -- *) Read Edit(configs/posttrain/secure-current.json) Edit(configs/posttrain/experiments/*.json) Write(configs/posttrain/experiments/*.json) Edit(docs/experiments/*.md) Write(docs/experiments/*.md) Glob(configs/posttrain/experiments/*.json) Grep
disable-model-invocation: true
---
Capture `BASE_REF` once. For each serial round, edit only strict experiment files, choose exactly one lever, then invoke:

```bash
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref "$BASE_REF" --budget-minutes 90 --promote
```

Never use redirections or general-purpose file mutation commands. Stop on integrity or technical failure and after two non-improving rounds.
