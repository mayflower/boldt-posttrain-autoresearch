---
description: Discover and prepare verified German training data
argument-hint: "dry|real"
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli data *) Read
disable-model-invocation: true
---
Use the selected mode consistently:

```bash
uv run --locked python -m boldt_posttrain.cli data discover --real --config configs/posttrain/current.json
uv run --locked python -m boldt_posttrain.cli data prepare --real --config configs/posttrain/current.json
```

For plans use `--dry-run` on both commands. Emit each JSON result unchanged.
