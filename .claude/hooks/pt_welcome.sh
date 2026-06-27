#!/usr/bin/env bash
# SessionStart hook: short intro for the post-training AutoResearch loop.
python3 - <<'PY'
import json
msg = """\
PostTrain AutoResearch loop — Boldt DC 1B German

This repo is instrumented for iterative German post-training of mayflowergmbh/boldt-dc-1b-german-it-16k-dpo using German OpenEuroLLM data.

Start with:
/pt-orient        rules, repo/script readiness, current state
/pt-data dry      discover German OpenEuroLLM candidates without downloading full datasets
/pt-baseline dry  check baseline-eval plumbing
/pt-run 3 dry     run the autonomous dry loop

Real runs require implemented scripts, GPU access, clean data manifest, and explicit real mode.
Full rules: AUTORESEARCH_POSTTRAIN.md.
"""
print(json.dumps({"systemMessage": msg}))
PY
