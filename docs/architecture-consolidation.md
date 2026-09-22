# Architecture consolidation plan

The repository runs two parallel systems under one CLI. This document maps them
precisely and sets the order in which they should be merged. It is the plan the
audit asked for ("ein gemeinsames Run-Format, eine Kandidatenauflösung, eine
Evaluation und ein Scoring ... die bestehenden Erzeuger und Verbraucher
zusammenführen").

## The two systems

| Concern | System A ("recipe", top-level) | System B ("secure", canonical) |
|---|---|---|
| Config schema | `configs/posttrain/current.json` (`extends` base.json, `recipe-policy.json`, `org`, `sources[].dataset/configs/splits`) | `configs/posttrain/secure-current.json` (`schema_version`, `policy.json`, `sources[].dataset_id/config/split`) |
| Config loader | `config.resolve_config` (deep-merge `extends`) | `secure_compat.config.load_experiment` |
| Training producer | `training.run_training_trial` → `_train_real` | `secure_compat.training.train_adapter` |
| Run id | `{specialist}-{kind}-{real|dry}-{stamp}` (no microseconds, no random suffix) | `artifacts.new_run_id` → matches `RUN_ID_RE` |
| Run card | `provenance.new_run_card` (`model` is a string; own `validate_run_card`) | `artifacts` run card (`model` is a structured dict; event-chained) |
| Event chain | none | `artifacts.EventLog` (`events.jsonl` + head) |
| Evaluation | `evaluation.run_real_evaluation` (dev/proxy, `_default_validate` + top-level scorers) | `secure_compat.evaluation._publish_evaluation` (`score_output`, full validators) |
| Scoring | `scoring` top-level defs | `secure_compat.scoring.create_score` |
| Frontier | `frontier` top-level defs | `secure_compat.frontier` |
| Candidate resolution | — (produces cards the resolver rejects) | `resolver.resolve_candidate` |
| CLI entry points | `train`, `eval run`, `baseline`, `score`, `promote` | `loop run`, `distill` |

The top-level modules (`config`, `data_pipeline`, `distillation`, `evaluation`,
`frontier`, `merge`, `preference`, `provenance`, `scoring`, `training`) are each
a partial second implementation **plus** a re-export of the secure twin at the
end of the file. `secure_compat` in turn imports the shared leaves back from the
top level (`artifacts`, `policy`, `resolver`, `verifiers`).

## Why this is a correctness problem, not only duplication

- **P0: manual candidates cannot be evaluated.** `resolver.resolve_candidate`
  (used by `eval run --candidate`, `score`, `promote`) validates cards with
  `artifacts.validate_run_card` and requires a structured `model` dict, a
  canonical run id, and a successful entry in the hash-chained event log.
  `run_training_trial` writes none of these, so the documented
  "train → evaluate the returned id → score → promote" chain is broken. Only the
  `loop`, which uses `train_adapter`, connects end to end.
- **The config switch hides what ran.** `evaluation._publish_evaluation` rewrites
  `current.json` to `secure-current.json` by filename and monkeypatches
  `generate_cases`/`run_lm_eval` into `secure_compat` at call time. It exists
  because the two config schemas are incompatible and the tests
  (`tests/artifact_chain.py`, `tests/test_baseline.py`) feed the recipe schema
  into the secure evaluator. It is load-bearing today, which is exactly why the
  schemas must merge before it can go.

## Canonical target

**System B (secure) is canonical.** It is the connected, integrity-bearing path:
canonical run ids, event chain, structured model provenance, real validators,
and a resolver that consumes its own producer's output. System A is the earlier
"recipe" path; its distinguishing artifacts (recipe run ids, string `model`,
no events) are the source of the P0. Consolidation means routing System A's CLI
verbs onto System B's producers and retiring the recipe duplicates -- not adding
a translator between them.

## Staged plan (each stage keeps the suite green and integrity PASS)

1. **Config schema.** Make `config.resolve_config` accept the secure schema (or
   convert `current.json` to it once, at the edge) so a single experiment schema
   feeds both `train_adapter` and `run_real_evaluation`. Update the ~10 tests and
   `tests/artifact_chain.py` that pin the recipe schema. Removing this dependency
   is what later lets the `_publish_evaluation` filename switch disappear.
2. **Training producer.** Route `train sft|cpt|preference` (and the three
   `scripts/pt_train_*.py`) through the loop's single-lever producer so a manual
   run writes a canonical, event-chained, resolver-compatible card. Delete
   `run_training_trial` and the recipe run-card path. This closes the P0.
   Requires a GPU end-to-end check (train → resolve → eval → score) mirroring
   `tests/test_training_gpu.py`, because the manual real path has no coverage
   today.
3. **Evaluation.** Point `eval run`/`baseline` at the secure evaluator with the
   unified schema; delete `run_real_evaluation` and the `_publish_evaluation`
   filename switch and monkeypatch.
4. **Scoring and frontier.** Collapse the top-level `scoring`/`frontier` defs
   into their secure twins; keep one `create_score`, one frontier pointer.
5. **Leaf cleanup.** Remove the now-empty top-level shims, leaving `secure_compat`
   as the implementation and the top-level names as thin, honest imports (or move
   `secure_compat` up and drop the package split entirely).

## Verification gates

Every stage: `ruff check`, `ruff format --check`, full `pytest`, and
`scripts/check_posttrain_integrity.py` PASS on a clean tree. Stage 2 additionally
requires a GPU run proving a manually trained candidate resolves and scores,
since that path is currently untested — which is how the P0 survived.

## Empirical findings during execution (2026-09-22)

Working the plan surfaced a stronger coupling than the idealized 1→5 order assumed:

- **The recipe config schema falls last, not first.** `current.json`'s recipe
  fields exist only because the manual CLI verbs (`train`, `eval run`, `baseline`,
  `score`) still read them. The schema cannot be removed until those verbs are
  rerouted onto the secure producers. So the true order is reroute-then-drop, and
  "Stage 1: config" is really the closing cleanup.
- **Only the training reroute is GPU-gated.** Evaluation and scoring run on CPU
  (the suite proves it via `tests/artifact_chain.py`). Rerouting `eval`/`score`
  is CPU-verifiable; rerouting `train` (Stage 2) needs a GPU end-to-end run
  because the manual real training path has no coverage today.
- **Two live scorers.** Manual `score` uses top-level `scoring.score_run`
  (compares two saved summaries); the loop uses secure `create_score`
  (event-chained). Collapsing them is coupled to the eval reroute.

### Done

- **Config filename switch removed.** `evaluation._publish_evaluation` no longer
  rewrites `current.json` to `secure-current.json` by filename; callers state the
  config explicitly (`tests/artifact_chain.py`, `tests/test_baseline.py` updated).
  The only remaining seam is forwarding a test-overridden `run_lm_eval` for
  offline stubbing. Full suite green.

### Handed back (GPU-gated or high-risk to land blind)

- Training producer reroute (Stage 2) and the config-schema removal it unblocks.
  These change what runs and must be verified on the target GPU, not asserted.

## Execution status (2026-09-22, branch refactor/consolidate-systems)

Done and CPU-green (201 passing) on the branch:
- **3a** config filename switch removed (`81b705d`).
- **2 (code)** manual `train sft|cpt|preference` now runs the loop's single secure
  producer via `loop.train_one_lever`; `run_training_trial` and the three recipe
  training scripts are gone; a latent `verify_data_manifest` signature crash in
  the loop was fixed (`3c0bdc4`).
- **HF alignment** `hf-transfer` is a locked dep, auto-enabled on import (`826961f`).

Blocked, not done:
- **GPU end-to-end of stage 2.** The host's egress to HuggingFace's file backend
  stalls the client on large files (plain / hf-transfer / HF_HUB_DISABLE_XET all
  move 0 bytes; curl gets ~8 MB/s). Deferred until the network is fixed, per
  decision; then the HF-idiomatic path runs unchanged.

Remaining structural prerequisite before a documented end-to-end run works, even
with networking fixed:
- **Secure data + baseline CLI entries are unwired.** `data discover` / `data
  prepare` still forward to the recipe scripts and write a recipe manifest, but
  `train_one_lever` (and the loop) consume the secure manifest written by
  `secure_compat.data_pipeline.run_cli` (the `data/current.json` pointer + event
  chain), which no CLI verb invokes. Wire `_data_discover_command` /
  `_data_prepare_command` (and baseline) to the secure pipeline before the run.
  This is CPU-verifiable and independent of the network.

### Runbook once networking is fixed and the secure data entries are wired

```bash
uv run --locked python -m boldt_posttrain.cli data discover --real --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli data prepare  --real --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli baseline run  --real --allow-gpu --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli train sft --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
# train prints the candidate run id; it now resolves:
uv run --locked python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate <run_id>
uv run --locked python -m boldt_posttrain.cli score --candidate <eval_run_id>
```
