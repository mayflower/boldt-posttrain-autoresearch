# Boldt Post-Training AutoResearch

Reproducible German-first post-training for the revision-pinned seed
`mayflowergmbh/boldt-dc-1b-german-it-16k-dpo@a24720616fc0ae0d0e8d2009d1c4eddec56fd15c`.
The system discovers and materializes licensed German data, trains real PEFT adapters, evaluates
the exact requested candidate, scores a cryptographically linked artifact chain, and updates an
immutable frontier only after default-deny integrity succeeds.

## Install

Linux, Python 3.10+, and `uv` are required. Real training targets NVIDIA CUDA; the production
QLoRA profile supports a single 48-GB GPU.

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
export CUDA_VISIBLE_DEVICES=0
scripts/sync_env.sh
uv run --locked python -m boldt_posttrain.cli policy validate
uv run --locked python -m boldt_posttrain.cli doctor --mode all
```

`uv` owns the environment. `uv.lock` is the single source of truth, `--locked` refuses to
re-resolve it silently, and every documented command runs through `uv run --locked` so no
invocation depends on a previously activated shell. There is no Conda environment and no
hand-managed `.venv` to activate. `pip install -e '.[train,data,eval,merge]'` still works but does
not replace the lock.

`uv run` also puts `.venv/bin` on `PATH`, which the code relies on: the merge lever resolves
`mergekit-yaml` and the evaluation lever resolves `lm-eval` via `shutil.which`. Calling
`.venv/bin/python` directly skips that and makes `doctor --real` fail with an unavailable Mergekit.

With the default order `FASTEST_FIRST`, device `0` on the reference host is the 48-GB
NVIDIA RTX A6000.

## Modes

Every mutating operation requires exactly one of `--dry-run` or `--real`. Discovery plans are
written under `outputs/posttrain/plans/`; other dry runs write only explicitly labelled dry-run
artifacts. Training and distillation additionally require
`--allow-gpu --allow-checkpoints`; evaluation requires `--allow-gpu`; merge requires
`--allow-checkpoints` and uses `--allow-gpu` for its configured GPU path. No command falls back to
CPU, another model, another trainer, or a smaller benchmark.

## Workflow

```bash
uv run --locked python -m boldt_posttrain.cli data discover --real --config configs/posttrain/current.json
uv run --locked python -m boldt_posttrain.cli data prepare --real --config configs/posttrain/current.json
uv run --locked python -m boldt_posttrain.cli baseline run --real --allow-gpu --config configs/posttrain/current.json
uv run --locked python -m boldt_posttrain.cli train sft --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate train-sft-20260721T120000.000000Z-0123456789abcdef
uv run --locked python -m boldt_posttrain.cli score --candidate eval-20260721T130000.000000Z-0123456789abcdef
uv run --locked python -m boldt_posttrain.cli promote --candidate train-sft-20260721T120000.000000Z-0123456789abcdef --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247
```

Other real levers:

```bash
uv run --locked python -m boldt_posttrain.cli train cpt --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train preference --method dpo --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train preference --method kto --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train preference --method orpo --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli distill --teacher mayflowergmbh/boldt-dc-1b-german-it-16k-dpo@a24720616fc0ae0d0e8d2009d1c4eddec56fd15c --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli merge search --real --allow-gpu --allow-checkpoints --config configs/posttrain/current.json --budget-minutes 90
```

One deterministic experiment round:

```bash
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247 --budget-minutes 90
```

## Trust Model

- Human-owned rules live only in `configs/posttrain/policy.json`; experiment files cannot override
  model revisions, licenses, tasks, thresholds, scoring, promotion, or integrity.
- Run cards are schema v1 and hash every relevant input/output. Run IDs contain UTC microseconds
  and 128 random bits.
- `events.jsonl` is hash-chained and anchored by `events.head.json`; successful events include the
  exact run-card hash.
- Baseline, score, promotion history, and frontier pointers are immutable or atomically replaced
  under locks. A free `summary.json` has no authority.
- The local boundary detects modification, replacement, and truncation. An attacker able to
  rewrite code, policy, Git history, event log, and head together is outside this boundary.

## Exit Codes

- `0`: operation succeeded and its gate passed
- `1`: operation succeeded technically, candidate rejected
- `2`: invalid CLI or experiment configuration
- `3`: missing prerequisite or dependency
- `4`: execution failure
- `5`: integrity or manipulation failure

See `docs/operations.md`, `docs/evaluation.md`, `docs/data-pipeline.md`, `docs/training.md`,
`docs/preference-and-distillation.md`, `docs/scoring-and-promotion.md`, and
`docs/merge-search.md`.
