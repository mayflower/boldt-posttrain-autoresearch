# Prompt-pack implementation report

Date: 2026-08-13

## Result

W01–W05 are implemented as one serial, artifact-derived workflow. Productive paths perform real
discovery, preparation, training, mergekit merging, local generation, lm-eval evaluation, scoring,
failure mining, verified synthesis, mix probes, Successive Halving, and RLOO. No commit or push was
performed.

## Main changed surfaces

- Unified commands and bootstrap: `src/boldt_posttrain/cli.py`, `bootstrap.py`, and production slash
  commands.
- Data and evaluation: streaming bounded shards, SQLite MinHash near-deduplication, exact hashed
  FastText language ID, complete decontamination, proxy/dev/promotion profiles, immutable baselines,
  paired confidence intervals, and protected raw generations.
- Training and search: conversational DPO/KTO/ORPO, default LoRA/rsLoRA/PiSSA, explicit device
  capability gates, streaming SFT, serial 64/192/remainder Successive Halving, specialist frontiers,
  and measured mix plans.
- Improvement paths: fixed failure taxonomy, mechanical Best-of-N synthesis, fixed reward registry,
  PEFT RLOO, efficiency comparisons, and opt-in Liger measurement.
- Provenance: hashed artifacts, hash-linked events, run-card cost metrics, score/profile checks, and
  verified frontier inputs.
- Reproducibility: exact optional dependency versions in `pyproject.toml` and `uv.lock`, plus a
  manually dispatched 48-GB self-hosted GPU acceptance workflow.

## Removed or simplified old paths

- Replaced the unsupported default `data` lever with executable `sft` without rewriting it during
  bootstrap.
- Removed exact-one-merge-candidate assumptions; every candidate gets proxy evaluation and only the
  winner gets development evaluation.
- Replaced flat preference strings and training-row-derived probes with the protected conversational
  representation and probe set.
- Removed fixed `cuda:0`/compute-capability assumptions in favor of explicit devices and measured
  CUDA, VRAM, BF16, and bitsandbytes capabilities.
- Removed silent scoring/config fallbacks and stale slash-command instructions that created stubs.
- Kept one local artifact/event system and one serial scheduler; no service, queue, parallel branch,
  or alternate optimization framework was introduced.

## Verification evidence

| Command or proof | Exit | Result |
|---|---:|---|
| `uv sync --extra train --extra eval --extra data --extra merge --extra dev --frozen` | 0 | Locked environment installed |
| `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -rs` | 0 | 117 passed in 97.94 s |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | 67 files formatted |
| `uv run python -m compileall -q src scripts tests` | 0 | Import/bytecode smoke passed |
| `uv run python -m boldt_posttrain.cli policy validate` | 0 | Protected policy valid |
| `uv run python -m boldt_posttrain.cli eval validate-suite` | 0 | 96 cases; suite hash `d8a0396c2a4687ff3294107abe748c9b0f26f33390c0d40d2857e3d0c964d796` |
| `git diff --check` | 0 | No whitespace errors |
| Both required forbidden-pattern searches | 1 | Expected: no matches |

The test suite includes these real integration proofs:

- local Tiny Causal LM generation plus real base-model and PEFT-adapter lm-eval subprocesses,
  summary, and score;
- one optimizer step for conversational DPO, KTO, and ORPO;
- optimizer-step/save/reload/forward for default LoRA, rsLoRA, and `pissa_niter_4`;
- Tiny RLOO optimizer execution, adapter save/reload/forward, reward-error propagation, and metrics;
- real mergekit subprocesses for linear, SLERP, TIES, and DARE-TIES over two local PEFT adapters;
- six serial trials reducing 6 → 2 → 1 at cumulative 64 and 256 steps;
- Best-of-4 mechanical synthesis, no eval-prompt leakage, three-source mix probes, and no implicit
  source repetition;
- tamper detection for manifests, shards, decontamination, events, summaries, and frontiers.

## Measured streaming effect

The final code processed the same deterministic 100,000-row fixture twice:

| Run | Runtime | Written rows | Shards | Peak RSS delta | Manifest hash |
|---|---:|---:|---:|---:|---|
| 1 | 27.84 s | 99,992 | 3 | 37,296 KiB | `b785e6b4f239684663c00b79feefaba24b9e902c90625f6ead37f14eb7479a54` |
| 2 | 27.44 s | 99,992 | 3 | 37,728 KiB | `b785e6b4f239684663c00b79feefaba24b9e902c90625f6ead37f14eb7479a54` |

Rows were generated lazily; the materializer did not retain a Python dictionary per complete row.
The eight removed rows were deterministic near-duplicates. Shards stayed below 50,000 rows and 128
MiB, and the temporary SQLite index was removed after atomic publication.

## Promptkit2 real acceptance

The canonical conversational source is `allenai/Dolci-Instruct-SFT` at immutable revision
`bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` (ODC-BY). It is explicitly allowlisted by dataset,
revision, and license. German FastText filtering runs before bounded hash sampling. The requested
gated translated mirror remained unavailable and its flat response-only schema is not used to
invent prompts.

Two complete preparation passes scanned 2,152,112 rows and produced identical manifest hash
`f3b831315dc8dda8bef22730e9a13bd5722f21ccc221c1ff8379a1c580b59704`, quality reports, and shard
hashes. Of 1,357 confidently German conversations, structural filtering, protected-suite
decontamination, and near-deduplication left 116 clean SFT rows (114 train, 2 validation). Exact
chat-template measurements were p50 69.5, p95 212.75, p99 418.6 tokens, zero truncation, zero empty
assistant masks, and 0.46947 assistant-supervision fraction.

On the isolated RTX A6000, equal 10-step QLoRA runs all saved, reloaded, and evaluated every step:

| Run | Best validation loss | Tokens/s | Peak allocated VRAM | Result |
|---|---:|---:|---:|---|
| full-sequence control `dolci-full-loss-specialist-real-20260813T105551` | 2.22254 | 54.48 | 1,401,605,120 B | reload passed |
| assistant-only `dolci-assistant-loss-specialist-real-20260813T105637` | 1.75462 | 55.42 | 1,401,605,120 B | reload passed |
| Liger `dolci-liger-specialist-real-20260813T113748` | 2.23175 | 53.48 | 1,319,763,456 B | below enable threshold |
| rsLoRA `dolci-rslora-specialist-real-20260813T113837` | 2.15190 | 55.29 | 1,401,605,120 B | research candidate only |

The technically clean proxy evaluation tied four of five lm-eval scores. The full-sequence control
slightly led TruthfulQA (0.42351 versus 0.42301), format following (1.0 versus 0.0), and German
retention (1.0 versus 0.875). Both failed the safety hard gate, so neither run was promoted. A full
development lm-eval was stopped as non-promotional when its five-task workload could not complete
comfortably within the shared-GPU acceptance budget; no partial score was reported.
