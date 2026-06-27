---
description: Run merge search over complementary Boldt specialists and evaluate merged candidates
argument-hint: "[dry|real]"
allowed-tools: Bash(python scripts/pt_merge_search.py *) Bash(python scripts/pt_eval.py *) Bash(python scripts/pt_score.py *) Bash(python scripts/pt_frontier_status.py *) Bash(python scripts/check_posttrain_integrity.py *) Bash(cat *) Bash(find *) Read Edit
disable-model-invocation: true
---
# PostTrain AutoResearch — merge search

Parse `$ARGUMENTS`: mode default `dry`. Set `MODE_FLAG=--dry-run` or `MODE_FLAG=--real` before running commands.

Inspect eligible candidates:

```bash
find outputs/posttrain/runs -name run_card.json -maxdepth 4 2>/dev/null | sort || true
test -f scripts/pt_frontier_status.py && python scripts/pt_frontier_status.py --format markdown || true
```

Run merge search:

```bash
python scripts/pt_merge_search.py \
  --config configs/posttrain/current.json \
  --runs outputs/posttrain/runs \
  --out outputs/posttrain/merge \
  "$MODE_FLAG"
```

If candidates are produced, evaluate the best/newest merge:

```bash
python scripts/pt_eval.py \
  --config configs/posttrain/current.json \
  --candidate latest-merge \
  --out outputs/posttrain/evals \
  "$MODE_FLAG" || true

python scripts/pt_score.py \
  --config configs/posttrain/current.json \
  --candidate latest-merge \
  --out outputs/posttrain/score-latest.json \
  "$MODE_FLAG" || true
```

Run integrity. Report: parents, method, parameters, eval artifact, score/gates, keep/reject.
