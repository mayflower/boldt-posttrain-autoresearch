---
description: Inspect post-training readiness without mutation
argument-hint: ""
allowed-tools: Bash(uv run --locked python -m boldt_posttrain.cli status) Bash(uv run --locked python -m boldt_posttrain.cli policy validate) Bash(uv run --locked git status --short) Read Glob Grep
disable-model-invocation: true
---
Read the operating documents, validate the policy and run status. Report `readiness`
(data manifest, baseline, `loop_ready`) and its `next_command`. Do not edit files.
