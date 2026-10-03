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
uv run --locked bash scripts/sync_env.sh
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
orientation with the next steps. On the first start in a new checkout, accept the trust prompt
so the project hooks in `.claude/settings.json` are active.

### A typical session

```text
/pt-orient          validate the policy, show verified status, name the next exact command (read-only)
/pt-data dry        plan data discovery and preparation; then /pt-data real (once)
/pt-baseline real   create the immutable seed baseline (once, GPU)
/pt-run 3 real      let the agent run up to 3 serial research rounds
/pt-status          verified status, frontier and report
```

What `/pt-run N real` does per round: Claude Code captures the base Git ref once, writes its
hypothesis and exactly one lever (`sft`, `cpt`, `preference`, `grpo`, `rlvr`, `opd`, `sdpo`,
`sdft`, `distill` or `merge`) into `configs/posttrain/secure-current.json` (plus an optional
`configs/posttrain/experiments/*.json` file and a note in `docs/experiments/`), and invokes:

```bash
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref "$BASE_REF" --budget-minutes 90 --promote
```

It then reads the verdict and starts the next round. It stops after N rounds, on any technical or
integrity failure (nonzero exit), or after two consecutive rounds without a passing improvement.

Further commands for single steps:

| Command | Purpose |
| --- | --- |
| `/pt-train dry\|real sft\|cpt\|preference` | one training job |
| `/pt-rlvr` | RLOO online RL with mechanical rewards |
| `/pt-eval dry\|real <run-id>` | evaluate one exact candidate |
| `/pt-trial dry\|real <run-id>` | evaluate and score one exact candidate |
| `/pt-merge dry\|real` | merge search over scored candidates |
| `/pt-search <search-config.json>` | serial Successive Halving search |
| `/pt-failures <dev-eval-run-id>` | failure statistics from a development evaluation |
| `/pt-integrity --base-ref REF` | default-deny integrity gate |
| `/pt-promote <run-id> <base-ref>` | promote one verified candidate |

The `/pt-*` commands are human-invoked only (`disable-model-invocation`): Claude cannot trigger
them itself. Each command pre-authorizes exactly the CLI calls it needs. A CLI call Claude makes
directly through Bash is allowed by the guard but still goes through the normal permission prompt,
so do not bypass permissions if you want to approve real runs yourself. Slash commands also work non-interactively, for
example `claude -p "/pt-orient"`. Long `/pt-run` sessions are best kept in an interactive
session (e.g. inside `tmux`) so the verdicts stay visible.

### What the agent can and cannot do

A PreToolUse hook (`.claude/hooks/guard_posttrain.py`) enforces the trust boundary for the
`Bash`, `Edit`, `Write`, `Read`, `Glob` and `Grep` tools:

- Writes only to `configs/posttrain/secure-current.json`, `configs/posttrain/experiments/*.json`
  and `docs/experiments/*.md`. Policy, scorer, evaluation data, source code, baselines and
  runtime artifacts are blocked.
- Shell only for `uv run --locked python -m boldt_posttrain.cli …`, `uv run --locked git rev-parse
  HEAD`, `uv run --locked git status --short` and `uv run --locked git diff -- …`, each as a single
  command: no `&&`, `;`, pipes, redirections or substitutions.
- Read, Glob and Grep are unrestricted.

A blocked call is denied with the reason, and Claude adapts. Expect to see this when the agent
first tries a chained shell command. Tools outside that list (for example `NotebookEdit` or MCP
tools) are not covered by the hook; keep them unapproved during research sessions.

### Working on this repository with Claude Code

The guard also blocks ordinary development (editing `src/`, tests, docs). To work on the code
itself, disable the hooks locally with `.claude/settings.local.json`:

```json
{ "disableAllHooks": true }
```

Delete the file before any research session. It is not ignored by Git, so `git status` shows it,
but the integrity gate does not check `.claude/`; a forgotten override silently removes the
trust boundary.

### Other agents and manual use

Codex and other agents read `AGENTS.md` directly and run the same `uv run --locked …` commands.
They are not covered by the Claude Code hook; the CLI's own policy, integrity and promotion gates
still apply. The commands below also work by hand without any agent.

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
uv run --locked python -m boldt_posttrain.cli promote --candidate train-sft-20260721T120000.000000Z-0123456789abcdef --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247
```

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
```

Before running another lever, configure its data and parameters in the strict secure config.
GRPO/RLVR need verified rows, SDPO needs feedback, SDFT needs demonstrations, and OPD needs
a distinct, licensed teacher at an exact revision. `distill` aliases online OPD.
Merge inputs must name exact, scored candidate run IDs in the config.

One deterministic experiment round:

```bash
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247 --budget-minutes 90 --promote
```

`status` and `report` return verified JSON from the canonical event chain and pointers.

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
