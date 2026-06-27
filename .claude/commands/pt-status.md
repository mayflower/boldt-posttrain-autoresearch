---
description: Read-only status of data manifest, run cards, evals, frontier, and current config
argument-hint: ""
allowed-tools: Bash(cat *) Bash(tail *) Bash(ls *) Bash(find *) Bash(python scripts/pt_status.py *) Bash(python scripts/pt_report.py *) Read
disable-model-invocation: true
---
# PostTrain AutoResearch — status

Read current state only. Do not edit.

Run what exists:

```bash
cat configs/posttrain/current.json 2>/dev/null || echo "(missing current config)"
tail -n 20 outputs/posttrain/results.tsv 2>/dev/null || echo "(no results.tsv)"
cat outputs/posttrain/frontier.json 2>/dev/null || echo "(no frontier.json)"
find outputs/posttrain -maxdepth 3 -type f \( -name 'run_card.json' -o -name 'summary.json' -o -name 'manifest.json' \) 2>/dev/null | sort | tail -n 30 || true
test -f scripts/pt_status.py && python scripts/pt_status.py --format markdown || true
test -f scripts/pt_report.py && python scripts/pt_report.py --format markdown --no-write || true
```

Summarize:

- data readiness;
- baseline readiness;
- current best candidate, if any;
- latest failed gates;
- next highest-value lever.
