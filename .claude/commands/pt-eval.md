---
description: Evaluate one exact model or candidate run
argument-hint: "dry|real <exact-run-id>"
allowed-tools: Bash(python -m boldt_posttrain.cli eval run *) Read
disable-model-invocation: true
---
Candidate identifiers are exact run IDs; there is no latest alias.

```bash
python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate "$CANDIDATE"
```

For a plan use `--dry-run`. Emit JSON unchanged and stop on nonzero exit.
