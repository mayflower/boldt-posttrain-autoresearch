# Boldt German Post-Training — Claude Code Instructions

## Mission

Improve `mayflowergmbh/boldt-dc-1b-german-it-16k-dpo` as a German-first small instruction model using only auditable German subsets of `openeurollm/*` datasets plus explicitly allowed local/evaluation data. The research style is iterative: train complementary small specialists, merge them, evaluate regressions, and promote only measured improvements.

## Architecture and repository layout

This repo is an **orchestration/scaffolding pack**, not a conventional code project. It encodes an auditable post-training research loop as four layers:

- `.claude/commands/pt-*.md` — slash commands that are the operator/agent interface. Each is `disable-model-invocation: true` (must be invoked explicitly), has narrow `allowed-tools`, and calls Python scripts **by contract** rather than embedding logic.
- `scripts/pt_*.py` + `scripts/check_posttrain_integrity.py` — thin CLIs the commands call. Their I/O shapes, required artifacts, and common flags are specified in `docs/posttrain-script-contracts.md`. They are **dry-run-first**: dry mode is pure stdlib and writes the contracted artifacts as unmeasured plumbing; `--real` paths gate on `--allow-gpu` + the optional ML stack and **fail closed** (never fabricate metrics) where the concrete trainer/eval is not yet implemented.
- `src/boldt_posttrain/` — the shared, stdlib-only engine the scripts import so logic lives in one auditable place: `config.py` (`extends`/deep-merge + validation), `provenance.py` (git/env/run cards), `recipe.py` (metrics skeleton + run-dir provenance + the `--real` gate), `scoring.py` (**protected** — the deterministic score + fail-closed gates), `training.py` (shared train-lever skeleton), `frontier.py` (read-only frontier view), `cli.py` (console entry points).
- `configs/posttrain/` — `base.json` (protected defaults + the integrity globs that `check_posttrain_integrity.py` reads), `current.json` (the live experiment; `extends` base.json — readers must merge base then overlay current), `experiments/*.json` (named overlays preserving past hypotheses). The loop edits `current.json` and `experiments/*`.
- `pyproject.toml` / `tests/` — the core + all gates run on stdlib only; heavy ML is optional: `pip install -e '.[train,eval,merge,data]'` (only needed for `--real`). `tests/` is a stdlib `unittest` suite (`python -m unittest discover -s tests`) covering scoring gates, config inheritance, and integrity classification. There is **no Makefile**: the `/pt-*` slash commands invoking `python scripts/pt_*.py` are the only interface.
- `outputs/posttrain/` — **all run state.** The loop is stateless except for these on-disk artifacts (`data/manifest.json`, `baseline/summary.json`, `runs/<id>/run_card.json`, `evals/<label>/summary.json`, `frontier.json`, `results.tsv`). There is no hidden controller or private state file.

`AUTORESEARCH_POSTTRAIN.md` is the full operating manual (data policy, lever menu, specialist taxonomy, merge policy, eval gate, scoring formula). Read it before changing loop behavior.

The research method: branch from the seed model → train small LoRA/QLoRA specialists → merge complementary checkpoints (mergekit) → evaluate against the German-core suite → promote only measured wins. `/pt-run` is the only autonomous loop; every other command is a single step.

## Slash-command interface and modes

Two execution modes apply to almost every command:
- `dry` (default): metadata/config/manifest/plan/integrity checks only. No GPU. **Dry-run metrics can never promote.**
- `real`: data prep, training, merge, eval. Scripts require explicit `--real` (plus `--allow-gpu`/`--allow-checkpoints` for training).

```
/pt-orient                     # read rules, report script + artifact readiness, suggest next step
/pt-bootstrap                  # (scripts already implemented) regenerate any missing stub from the contracts
/pt-status                     # read-only artifact summary
/pt-data [dry|real]            # discover + prepare German openeurollm subsets
/pt-baseline [dry|real]        # baseline eval of the seed model (required before training)
/pt-train [dry|real] <spec>    # train one specialist (general-de, reasoning-de, coding-de, safety-de, longcontext-de, raw-quality-de, preference-de)
/pt-merge [dry|real]           # merge search over eligible checkpoints
/pt-eval [dry|real] <label>    # evaluate a model/candidate
/pt-promote <label>            # promotion-gate check; writes frontier.json only if all gates pass
/pt-trial [dry|real] [lever]   # one full trial of a single lever
/pt-run [rounds] [dry|real]    # autonomous loop (default: 3 rounds, dry)
/pt-integrity [--base-ref REF] # protected-surface + git-cleanliness check
```

## Current repository state

- **All 15 canonical `scripts/pt_*.py` + `check_posttrain_integrity.py` are implemented** in the dry-run-first style above (modelled on the mature `boldt-embed-de` loop). The whole dry loop runs end to end on pure stdlib via the `/pt-*` commands (each calling `python scripts/pt_*.py`); the 16-test `unittest` suite passes.
- **`--real` is wired but intentionally fail-closed**: real data discovery/prepare, baseline/eval (German-core harness), training (LoRA/QLoRA/DPO/CPT), and merge (mergekit) require `--allow-gpu` + the optional extras and currently exit nonzero with an actionable message via `recipe.real_not_implemented(...)`. Implement each per `docs/posttrain-script-contracts.md` — and only there — when bringing real GPU runs online; never make a `--real` path fabricate metrics or a "clean" leakage/license status.
- **The repo is git-initialized** (branch `master`). On a fresh repo with nothing committed, `check_posttrain_integrity.py` flags every untracked file — including the protected `CLAUDE.md`/baseline — so it reports FAIL until there is a clean committed baseline; that is expected, not a bug.
- Quick validate: `python -m py_compile scripts/pt_*.py scripts/check_posttrain_integrity.py && python -m unittest discover -s tests && python scripts/pt_status.py --format markdown`. (The "Validation before any milestone claim" section below covers milestone-level checks.)

## Non-negotiable rules

- Inspect before editing. Keep changes small and reviewable.
- Do not commit model weights, checkpoints, Hugging Face caches, large datasets, API keys, W&B tokens, or private eval data.
- Never claim benchmark or model-quality improvements unless the command was run and its artifact exists under `outputs/posttrain/` with config, seed, git commit, data manifest, and evaluation summary.
- Training data and evaluation data must stay separate. Any contamination/leakage check that is missing, stale, or unparseable fails closed.
- German-first means: keep the Boldt chat template, preserve German output quality, penalize English bleed-through, and test German format-following explicitly.
- Dataset licensing/provenance must be visible in every run card. Unknown license means `training_usable: false` until reviewed.
- Prefer LoRA/QLoRA specialists and merge search over long sequential finetunes. The default path is branch → train specialist → merge → evaluate → promote, not one monolithic finetune.
- The base model family, tokenizer, chat tokens, and EOS settings are protected. Do not alter them silently.

## Protected surfaces

The AutoResearch loop may edit:

- `configs/posttrain/experiments/*.json`
- `configs/posttrain/current.json`
- new notes under `docs/experiments/`

The loop must not edit without explicit human request:

- evaluation scripts and benchmark manifests after baseline is established
- contamination/leakage scripts
- promotion gates/scoring scripts
- committed baseline reports under `outputs/posttrain/baseline/`
- root `CLAUDE.md`, `AUTORESEARCH_POSTTRAIN.md`, or release/model cards
- any file containing secrets, credentials, or dataset credentials

Run integrity checks before promotion and after any automated loop.

## Validation before any milestone claim

```bash
python scripts/pt_status.py --format markdown
python scripts/check_posttrain_integrity.py --format markdown
python scripts/pt_report.py --format markdown --no-write
```

For a real checkpoint claim, also require:

```bash
python scripts/pt_eval.py --model <candidate> --suite german-core --real --out outputs/posttrain/evals/<label>
python scripts/pt_promote.py --candidate <label> --format markdown
```

## Progress report format

Files changed · Commands run · Data used · Training artifact · Evaluation artifact · Gate verdict · Latest commit · Working tree · Risks.
