#!/usr/bin/env bash
# SessionStart hook: the same locked environment as the loop.
uv run --locked python - <<'PYWELCOME'
import json
print(json.dumps({"systemMessage": """PostTrain AutoResearch — Boldt DC 1B German

Read AGENTS.md, configs/posttrain/policy.json and AUTORESEARCH_POSTTRAIN.md.
/pt-orient         validate policy and inspect verified state
/pt-data dry       plan secure discovery/preparation
/pt-baseline dry   plan the immutable seed baseline
/pt-run 1 real     run one configured experiment through evaluation and scoring

Choose one lever in secure-current.json. Online distillation uses current student
rollouts; OPD requires a distinct licensed teacher. Stop on technical or integrity
failure. Real runs require explicit GPU/checkpoint permission and prerequisites.
"""}))
PYWELCOME
