---
description: Agentic outer loop over one deterministic real experiment at a time
argument-hint: "<rounds> real"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli *) Bash(uv run --locked git rev-parse HEAD) Bash(uv run --locked git status --short) Bash(uv run --locked git diff -- *) Read Edit(configs/posttrain/secure-current.json) Edit(configs/posttrain/experiments/*.json) Write(configs/posttrain/experiments/*.json) Edit(docs/experiments/*.md) Write(docs/experiments/*.md) Glob(configs/posttrain/experiments/*.json) Grep
disable-model-invocation: true
---
First run `uv run --locked python -m boldt_posttrain.cli status`. If `readiness.loop_ready` is
false, stop and report `readiness.next_command`; do not prepare data or baselines yourself.
Read the reports of earlier rounds (`uv run --locked python -m boldt_posttrain.cli report --loop <loop-id>`
for the loop IDs in status) before choosing the first hypothesis.

Capture `BASE_REF` once. For each serial round:

1. Before editing, tell the user in plain language: what the previous rounds showed (scores,
   failed gates, the metrics that moved), the hypothesis for this round and why it follows from
   that evidence, the lever, and every setting you change with old and new value.
2. Edit only strict experiment files (exactly one lever) and run:

```bash
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref "$BASE_REF" --budget-minutes 90 --promote
```

   Its progress lines (stage, training step/loss/ETA, evaluation progress, score, failed gates)
   go to stderr; relay the important ones while it runs.
3. Afterwards run `uv run --locked python -m boldt_posttrain.cli report --loop <loop-id>` and
   summarize it for the user: decision, score, each failed gate with value and threshold, the
   largest metric changes with their confidence intervals, training loss and time, and what you
   conclude. Write that summary to `docs/experiments/<loop-id>.md`.

The `seqkd` lever needs a `seqkd` block whose `generation_run` names a finished `/pt-seqkd`
run; if none exists, stop and report `/pt-seqkd real`. Never use redirections or general-purpose
file mutation commands. Stop on integrity or technical failure (exit 4) and after two
non-improving rounds; a rejected or `not_promoted` round (exit 1) is a result, not a failure.
Keep `training.max_steps` within the training time the round reports (about 64 s per step at the
current settings), so the learning-rate schedule completes before the budget stops training.
