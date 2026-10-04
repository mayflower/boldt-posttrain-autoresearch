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
export CUDA_VISIBLE_DEVICES=0   # evaluation runs on cuda:0; pick the physical GPU here
uv run --locked bash scripts/sync_env.sh
uv run --locked python -m boldt_posttrain.cli policy validate
uv run --locked python -m boldt_posttrain.cli doctor --mode all
```

`uv` owns the environment. `uv.lock` is the single source of truth, `--locked` refuses to
re-resolve it silently, and every documented command runs through `uv run --locked` so no
invocation depends on a previously activated shell. There is no Conda environment and no
hand-managed `.venv` to activate. `scripts/sync_env.sh` installs all extras and checks that CUDA,
`mergekit-yaml` and `lm-eval` are available.

`uv run` also puts `.venv/bin` on `PATH`, which the code relies on: the merge and evaluation levers
call `mergekit-yaml` and `lm-eval` from `PATH`. Calling `.venv/bin/python` directly skips that, so
merge and evaluation fail with a missing executable.

## How it is used: Claude Code drives, Python executes

The research loop runs inside a Claude Code session started in this repository. Nothing in the
code calls an LLM; the agent is the researcher, the CLI is the lab:

| Who | Does |
| --- | --- |
| Human | Owns `configs/posttrain/policy.json`, starts the session, authorizes GPU work by invoking the `/pt-*` commands. |
| Claude Code | Reads the contract, forms one hypothesis per round, records it as one lever in the strict experiment files, runs the loop, reads the verdict, decides the next round or stops. |
| `boldt_posttrain.cli` | Trains, resolves the candidate, evaluates, scores against the immutable baseline, and promotes only if every gate passes. Enforces the policy regardless of what the agent asks. |

### Start a session

```bash
cd boldt-posttrain-autoresearch
claude
```

On start, `CLAUDE.md` loads the agent contract `AGENTS.md`, and a session hook prints a short
orientation with the next steps.

### A typical session

```text
/pt-orient          readiness (data manifest, baseline) and the next command (read-only)
/pt-data real       discover and prepare verified training data
/pt-baseline real   create the seed baseline, or rebuild one that no longer verifies (GPU)
/pt-seqkd real      optional: teacher answers for the seqkd lever (GPU, 2-3 h)
/pt-run 3 real      let the agent run up to 3 serial research rounds
/pt-status          verified status, frontier and rounds
/pt-report <id>     report of one round: settings, training, metrics with CIs, gates, decision
```

What `/pt-run N real` does per round: Claude Code captures the base Git ref once, writes its
hypothesis and exactly one lever (`sft`, `cpt`, `preference`, `grpo`, `rlvr`, `opd`, `sdpo`,
`sdft`, `distill`, `seqkd` or `merge`) into `configs/posttrain/secure-current.json` (plus an optional
`configs/posttrain/experiments/*.json` file and a note in `docs/experiments/`), and invokes:

```bash
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref "$BASE_REF" --budget-minutes 90 --promote
```

While a round runs, timestamped progress lines on stderr show the stage, the applied training
settings, training step/loss/ETA, evaluation progress, lm-eval results, the score and every failed
gate. Each round writes `outputs/posttrain/loops/<loop-id>/report.md`, and
`pt report --loop <loop-id>` renders the same report for any round. The agent explains each
hypothesis before a round, summarizes the report afterwards and keeps that summary in
`docs/experiments/<loop-id>.md`. It then starts the next round. It stops after N rounds, on any technical or
integrity failure (nonzero exit), or after two consecutive rounds without a passing improvement.

Further commands for single steps:

| Command | Purpose |
| --- | --- |
| `/pt-train dry\|real sft\|cpt\|preference` | one training job |
| `/pt-rlvr` | RLOO online RL with mechanical rewards |
| `/pt-eval dry\|real <run-id>` | evaluate one exact candidate |
| `/pt-trial dry\|real <run-id>` | evaluate and score one exact candidate |
| `/pt-merge dry\|real` | merge search over scored candidates |
| `/pt-seqkd dry\|real` | teacher generation for sequence-level distillation |
| `/pt-integrity --base-ref REF` | default-deny integrity gate |
| `/pt-promote <run-id> <base-ref>` | promote one verified candidate |

The `/pt-*` commands are human-invoked only (`disable-model-invocation`): Claude cannot trigger
them itself. Each command pre-authorizes exactly the CLI calls it needs. A CLI call Claude makes
directly through Bash goes through the normal permission prompt, so do not bypass permissions if
you want to approve real runs yourself. Slash commands also work non-interactively, for
example `claude -p "/pt-orient"`. Long `/pt-run` sessions are best kept in an interactive
session (e.g. inside `tmux`) so the verdicts stay visible.

### What the agent may change

`AGENTS.md` limits the agent to `configs/posttrain/secure-current.json`,
`configs/posttrain/experiments/*.json` and `docs/experiments/*.md`. The integrity gate
(`pt integrity check`, run by every loop round and by promotion) fails a round whose changes
touch a protected surface listed in `policy.json`: policy, scorer, evaluation data and code,
baselines and the governance documents.

## Modes

Every mutating operation requires exactly one of `--dry-run` or `--real`. Data, evaluation and merge plans are written under `outputs/posttrain/plans/`;
training dry runs validate prerequisites without producing candidates. Training and distillation additionally require
`--allow-gpu --allow-checkpoints`; evaluation requires `--allow-gpu`; merge requires
`--allow-checkpoints` and uses `--allow-gpu` for its configured GPU path. No command falls back to
CPU, another model, another trainer, or a smaller benchmark.

## Manual CLI Workflow

```bash
uv run --locked python -m boldt_posttrain.cli data discover --real --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli data prepare --real --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli baseline run --real --allow-gpu --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli train sft --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate train-sft-20260721T120000.000000Z-0123456789abcdef
uv run --locked python -m boldt_posttrain.cli score --candidate eval-20260721T130000.000000Z-0123456789abcdef
uv run --locked python -m boldt_posttrain.cli promote --candidate train-sft-20260721T120000.000000Z-0123456789abcdef --base-ref "$BASE_REF"
```

Capture `BASE_REF=$(uv run --locked git rev-parse HEAD)` once before the first round; the
integrity gate vets every change since that commit. Run IDs are always the exact IDs returned by
the previous command.

Other real levers:

```bash
uv run --locked python -m boldt_posttrain.cli train cpt --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train preference --method dpo --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train preference --method kto --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train preference --method orpo --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train grpo --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train rlvr --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train opd --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train sdpo --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli train sdft --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli merge search --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli seqkd generate --real --allow-gpu --config configs/posttrain/experiments/seqkd-qwen3.8-27b-de.json
uv run --locked python -m boldt_posttrain.cli train seqkd --real --allow-gpu --allow-checkpoints --config configs/posttrain/experiments/seqkd-qwen3.8-27b-de.json --budget-minutes 90
```

Before running another lever, configure its data and parameters in the strict secure config.
GRPO/RLVR need verified rows, SDPO needs feedback, SDFT needs demonstrations, and OPD needs
a distinct, licensed teacher at an exact revision with the student's exact tokenizer.
`distill` aliases online OPD. Because no larger model shares Boldt's tokenizer, an external
teacher uses `seqkd` instead: `seqkd generate` lets a teacher from `policy.teachers`
(currently Qwen3.8-27B-FP8) answer prompts from the verified SFT manifest, filters the answers
through the same language, dedup and leakage gates as `data prepare`, and publishes them; the
`seqkd` lever then SFT-trains the student on the generation run pinned in
`seqkd.generation_run`. See `docs/preference-and-distillation.md`.
Merge inputs must name exact, scored candidate run IDs in the config.

One deterministic experiment round is the `loop run` command shown above for `/pt-run`.

`status` returns verified JSON from the canonical event chain and pointers.

## Trust Model

- Human-owned rules live only in `configs/posttrain/policy.json`; experiment files cannot override
  model revisions, licenses, tasks, thresholds, scoring, promotion, or integrity.
- Run cards are schema v1 and hash every relevant input/output. Run IDs contain UTC microseconds
  and 64 random bits.
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
