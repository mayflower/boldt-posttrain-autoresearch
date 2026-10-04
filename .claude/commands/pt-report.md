---
description: Show the human-readable report of one exact loop round
argument-hint: "<exact-loop-id>"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli report *) Bash(uv run --locked python -m boldt_posttrain.cli status) Read
disable-model-invocation: true
---
Loop IDs are exact; `status` lists them. Render the report and show it unchanged:

```bash
uv run --locked python -m boldt_posttrain.cli report --loop "$ARGUMENTS"
```

Then explain in a few sentences why the round passed or failed, citing the gate values.
