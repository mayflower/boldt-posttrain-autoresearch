# Boldt Post-Training AutoResearch

This repository implements a serial, single-GPU research loop for the German-first Boldt 1B seed.
It supports full-stream deterministic data sampling with manifest-level train/validation splits,
SFT/QLoRA/CPT and conversational preference training with best-checkpoint validation, merge search,
and a narrowly scoped GRPO operator for Math-Verify-compatible tasks.

Bootstrap a configured machine with:

```bash
uv sync --extra train --extra data --extra eval --extra merge --extra rl --frozen
uv run pt policy validate
uv run pt bootstrap run --real --allow-gpu --device cuda:0
```

The bootstrap derives `EMPTY → DISCOVERED → DATA_READY → BASELINE_READY → RESEARCH_READY` from
hashed artifacts. It writes no state database and never changes `configs/posttrain/current.json`.

Common productive commands are:

```bash
uv run pt search run --real --allow-gpu --allow-checkpoints --config <experiment.json>
uv run pt failures mine --eval-run <dev-run-id>
uv run pt data synthesize --from-failure-run <id> --teacher <model@revision> --real --allow-gpu
uv run pt mix probe --real --allow-gpu --allow-checkpoints
uv run pt train rlvr --real --allow-gpu --allow-checkpoints --budget-minutes 90
uv run pt train grpo --real --allow-gpu --allow-checkpoints --budget-minutes 90
```

SFT defaults to native TRL assistant-only loss. `training.use_liger_kernel` and
`training.use_rslora` are explicit experiment flags and remain disabled by default. CPT can
opt in to separately rate-scaled embedding/LM-head training through the `cpt` block, but
only on the non-quantized LoRA path. Unsloth is not an installed or supported backend.

Development, proxy, and promotion evaluations are separate. Promotion data is registered from an
absolute path outside the repository through `BOLDT_PROMOTION_SUITE`; only the baseline and
evaluation commands read it. A development score can guide search but cannot promote a model.

See [operations](docs/operations.md), [data pipeline](docs/data-pipeline.md),
[evaluation](docs/evaluation.md), [training](docs/training.md), and
[scoring/promotion](docs/scoring-and-promotion.md).
