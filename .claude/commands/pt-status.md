---
description: Read verified post-training status
argument-hint: ""
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli status) Read
disable-model-invocation: true
---
Run status and summarize `readiness`, the frontier and every loop round (status per loop ID).
Offer `/pt-report <loop-id>` for details. Treat legacy or unverifiable artifacts as untrusted.
