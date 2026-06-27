---
description: THE autonomous post-training AutoResearch loop — assess, choose one lever, run, eval, judge, repeat
argument-hint: "[rounds] [dry|real]"
allowed-tools: Bash(python scripts/pt_*.py *) Bash(python scripts/check_posttrain_integrity.py *) Bash(cat *) Bash(tail *) Bash(find *) Bash(git diff *) Bash(git status *) Read Edit Glob Grep
disable-model-invocation: true
---
# PostTrain AutoResearch loop

Run the loop yourself in this turn. There is no separate controller, no hidden state file, and no fixed step order. Artifacts on disk are the state.

Parse `$ARGUMENTS`:

- first integer token = rounds `N`, default `3`;
- mode = `dry` unless `real` is explicitly present.

## Objective

Improve `mayflowergmbh/boldt-dc-1b-german-it-16k-dpo` as a German-first small instruction model using German OpenEuroLLM data, without regressing German core benchmarks, safety, format-following, or language retention.

## Before round 1

Read:

- `AUTORESEARCH_POSTTRAIN.md`
- `CLAUDE.md`
- `configs/posttrain/current.json`
- latest artifacts under `outputs/posttrain/`

Run read-only state commands when available:

```bash
test -f scripts/pt_frontier_status.py && python scripts/pt_frontier_status.py --format markdown || true
test -f scripts/pt_report.py && python scripts/pt_report.py --format markdown --no-write || true
tail -n 20 outputs/posttrain/results.tsv 2>/dev/null || true
cat outputs/posttrain/frontier.json 2>/dev/null || true
```

## Each round k = 1..N

1. Assess measured state from artifacts.
2. Pick exactly one highest-value lever:
   - `data`: if no clean German OpenEuroLLM manifest exists or composition is weak;
   - `baseline`: if no baseline exists;
   - `sft-specialist`: if a capability gap is clear and clean SFT rows exist;
   - `pref-specialist`: if clean preference rows exist and response style needs alignment;
   - `cpt-specialist`: if high-quality German raw text can refresh knowledge/style without chat drift;
   - `merge`: if two or more complementary compatible candidates exist;
   - `distill`: if teacher/candidate comparisons exist and a sharpener is needed;
   - `eval`: if an unevaluated candidate exists;
   - `promote`: if a scored candidate appears promotable.
3. State the one-line rationale.
4. Run the tool for that lever in the selected mode. Real mode must pass explicit `--real`/`--allow-gpu`/`--allow-checkpoints` only where appropriate.
5. Evaluate any produced candidate.
6. Score/log if scripts exist.
7. Run integrity.
8. Decide `keep`, `reject`, `needs-real`, or `stop`.

## Stop early if

- a candidate passes promotion gate;
- two consecutive rounds bring no frontier improvement;
- integrity fails;
- required scripts/data are missing and `/pt-bootstrap` is the correct next move;
- real mode cannot proceed because clean data manifest or baseline is missing.

## Final report

Print a compact table:

`round · lever · rationale · commands/artifacts · verdict · best-so-far`

Then give the single recommended next command.

## Hard rules

- Dry-run metrics are plumbing only and can never promote.
- No quality claim without saved `outputs/posttrain/evals/<label>/summary.json` and run card.
- Unknown license/leakage/language status fails closed.
- Do not edit protected surfaces; if a protected edit happens, revert or stop.
