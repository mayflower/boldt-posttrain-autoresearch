---
description: Orient to the Boldt German post-training AutoResearch loop and summarize repo readiness
argument-hint: ""
allowed-tools: Bash(cat *) Bash(ls *) Bash(test *) Bash(git status *) Bash(python scripts/pt_status.py *) Bash(python scripts/check_posttrain_integrity.py *) Read Glob Grep
disable-model-invocation: true
---
# PostTrain AutoResearch — orient

Read `CLAUDE.md`, `AUTORESEARCH_POSTTRAIN.md`, `configs/posttrain/current.json`, and `docs/posttrain-script-contracts.md`.

Then inspect repo readiness:

```bash
ls -la .claude/commands configs/posttrain scripts 2>/dev/null || true
test -f scripts/pt_status.py && python scripts/pt_status.py --format markdown || true
test -f scripts/check_posttrain_integrity.py && python scripts/check_posttrain_integrity.py --format markdown || true
git status --short
```

Report:

- mission and non-negotiable rules in 5 bullets max;
- which required scripts exist/miss;
- whether outputs/baseline/data/frontier artifacts exist;
- the safest next slash command.

Do not edit files in this command.
