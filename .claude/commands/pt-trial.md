---
description: Run one post-training trial: choose or use a lever, execute, eval if possible, score, log, integrity
argument-hint: "[dry|real] [data|sft-specialist|pref-specialist|cpt-specialist|merge|distill|eval]"
allowed-tools: Bash(python scripts/pt_*.py *) Bash(python scripts/check_posttrain_integrity.py *) Bash(cat *) Bash(tail *) Bash(git diff *) Read Edit Glob Grep
disable-model-invocation: true
---
# PostTrain AutoResearch — one trial

Parse `$ARGUMENTS`:

- mode: `dry` unless `real` is present; set `MODE_FLAG=--dry-run` or `MODE_FLAG=--real` before running commands;
- lever: explicit lever if supplied, otherwise infer one from artifacts.

First inspect state:

```bash
test -f scripts/pt_frontier_status.py && python scripts/pt_frontier_status.py --format markdown || true
test -f scripts/pt_report.py && python scripts/pt_report.py --format markdown --no-write || true
tail -n 10 outputs/posttrain/results.tsv 2>/dev/null || true
```

Choose exactly one lever:

- `data`: run `/pt-data` logic.
- `sft-specialist`: run `scripts/pt_train_specialist.py`.
- `pref-specialist`: run `scripts/pt_train_preference.py`.
- `cpt-specialist`: run `scripts/pt_train_cpt.py`.
- `merge`: run `scripts/pt_merge_search.py`.
- `distill`: run `scripts/pt_distill_trial.py`.
- `eval`: run `scripts/pt_eval.py` on the newest candidate.

For training in real mode, require existing `outputs/posttrain/data/manifest.json` with clean status. For dry mode, ask scripts to validate/plumb only.

After the chosen lever, run scoring/logging if artifacts exist:

```bash
test -f scripts/pt_score.py && python scripts/pt_score.py --config configs/posttrain/current.json --out outputs/posttrain/score-latest.json "$MODE_FLAG" || true
test -f scripts/pt_log_result.py && python scripts/pt_log_result.py --config configs/posttrain/current.json --results outputs/posttrain/results.tsv || true
python scripts/check_posttrain_integrity.py --format json || true
```

Report a compact verdict: lever · rationale · command(s) · artifacts · score/gates · integrity · keep/reject/needs-real.
