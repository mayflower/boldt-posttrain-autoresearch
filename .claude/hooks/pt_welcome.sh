#!/usr/bin/env bash
# SessionStart hook: the same locked environment as the loop.
uv run --locked python - <<'PYWELCOME'
import json
print(json.dumps({"systemMessage": """PostTrain AutoResearch — Boldt DC 1B German

Read AGENTS.md, configs/posttrain/policy.json and AUTORESEARCH_POSTTRAIN.md.
/pt-orient            readiness and the next command
/pt-data real         discover and prepare verified training data
/pt-baseline real     create or rebuild the seed baseline (GPU)
/pt-seqkd real        optional: Qwen3.8 teacher answers for the seqkd lever (GPU, 2-3 h)
/pt-run <n> real      n serial research rounds: train, evaluate, score, promote
/pt-status            verified status, frontier and rounds
/pt-report <loop-id>  report of one round: settings, metrics, gates, decision

Stop on technical or integrity failure. Real runs require explicit GPU/checkpoint permission.
"""}))
PYWELCOME
