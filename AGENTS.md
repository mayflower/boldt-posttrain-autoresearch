# Boldt Post-Training — Agent Contract

Authoritative instructions for any agent (Claude Code, Codex, or otherwise) driving
the German post-training AutoResearch loop. Claude Code loads this via `CLAUDE.md`;
other tools read this file directly.

## Read first

- `configs/posttrain/policy.json` — the human-owned rules (model revision, licenses,
  tasks, thresholds, scoring, promotion, integrity). Never editable by an agent.
- `AUTORESEARCH_POSTTRAIN.md` — the operating policy for the loop.

## What an agent may change

Only the experiment surface:

- `configs/posttrain/secure-current.json` — the experiment the loop runs (data
  source, lever, training hyperparameters).
- `configs/posttrain/experiments/*.json` — strict search-trial experiment files.
- `docs/experiments/*.md` — experiment notes.

Never edit policy, the scorer, evaluation data, integrity/promotion code, baseline
pointers, source code, or runtime artifacts during autonomous research.

## Hard rules

- Every command runs through `uv run --locked`. The bare interpreter form is not an
  equivalent shorthand: it depends on an activated shell and leaves `.venv/bin` off
  `PATH`, where the merge and evaluation levers resolve `mergekit-yaml` and `lm-eval`.
  `--locked` never rewrites `uv.lock`; a disagreement with `pyproject.toml` is a stop.
- Use exact run IDs and exact Hub revisions. Never `latest`, moving refs, shell
  redirections, general-purpose mutation commands, hidden retries, reduced batches,
  reduced suites, CPU fallback, or alternate training methods.
- Preserve every nonzero exit code. Stop immediately on technical or integrity
  failure, and after two consecutive rounds without a passing improvement.
- Never claim GPU validation without actual target-hardware commands and their exit
  codes. Every claim cites the returned run ID plus its run card, checkpoint hash,
  data-manifest hash, suite hash, score ID, and event sequence.

## The loop

One outer round records exactly one candidate-producing lever (`sft`, `cpt`,
`preference`, `distill`, or `merge`) and invokes the loop, which trains, resolves the
candidate, evaluates it, scores it against the immutable baseline, and — only if the
gates pass — promotes it. Capture the base ref once before all rounds:

```bash
BASE_REF=$(git rev-parse HEAD)
uv run --locked python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --base-ref "$BASE_REF" --budget-minutes 90
```

Python executes the recorded experiment deterministically; hypothesis selection
remains with the agent.

## Command reference

Prerequisites (produce the secure data manifest and the immutable baseline once):

```bash
uv run --locked python -m boldt_posttrain.cli data prepare --real --config configs/posttrain/secure-current.json
uv run --locked python -m boldt_posttrain.cli baseline run --real --allow-gpu --config configs/posttrain/secure-current.json
```

Individual levers and inspection:

```bash
uv run --locked python -m boldt_posttrain.cli train sft --real --allow-gpu --allow-checkpoints --config configs/posttrain/secure-current.json --budget-minutes 90
uv run --locked python -m boldt_posttrain.cli eval run --real --allow-gpu --candidate <run_id>
uv run --locked python -m boldt_posttrain.cli score --candidate <eval_run_id>
uv run --locked python -m boldt_posttrain.cli policy validate
uv run --locked python -m boldt_posttrain.cli status
```

## Tooling notes

- Claude Code additionally exposes `/pt-*` slash commands (`.claude/commands/`) as
  thin wrappers over the commands above, plus a PreToolUse guard and a session hook
  (`.claude/hooks/`) that enforce the editable surface. These are conveniences, not a
  separate contract — the rules here are authoritative for every agent.
- Codex and other agents run the `uv run --locked …` commands directly.
