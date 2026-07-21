---
description: Discover and prepare verified German training data
argument-hint: "dry|real"
allowed-tools: Bash(python -m boldt_posttrain.cli data *) Read
disable-model-invocation: true
---
Use the selected mode consistently:

```bash
python -m boldt_posttrain.cli data discover --real --config configs/posttrain/current.json
python -m boldt_posttrain.cli data prepare --real --config configs/posttrain/current.json
```

For plans use `--dry-run` on both commands. Emit each JSON result unchanged.
