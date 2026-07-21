---
description: Promote one exact verified candidate
argument-hint: "<exact-run-id> <base-ref>"
allowed-tools: Bash(python -m boldt_posttrain.cli promote *) Read
disable-model-invocation: true
---

```bash
python -m boldt_posttrain.cli promote --candidate "$CANDIDATE" --base-ref "$BASE_REF"
```

Preserve the exit code and emit the verdict unchanged.
