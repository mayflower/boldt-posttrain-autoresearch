# Boldt Post-Training Agent Contract

Read `configs/posttrain/policy.json` and `AUTORESEARCH_POSTTRAIN.md` before an experiment. The
agent may edit only `configs/posttrain/current.json`, `configs/posttrain/experiments/*.json`, and
`docs/experiments/*.md`. It must never edit policy, scorer, evaluation data, integrity code,
promotion code, baseline pointers, source code, or runtime artifacts during autonomous research.

Use exact run IDs and exact Hub revisions. Never use `latest`, moving refs, shell redirections,
general-purpose mutation commands, hidden retries, reduced batches, reduced suites, CPU fallback,
or alternate training methods. Preserve every nonzero exit code.

One outer round chooses and records exactly one candidate-producing lever (`sft`, `cpt`,
`preference`, `distill`, or `merge`) and invokes:

```bash
python -m boldt_posttrain.cli loop run --real --allow-gpu --allow-checkpoints --config configs/posttrain/current.json --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247 --budget-minutes 90
```

Capture the base ref once before all rounds. Stop immediately on technical or integrity failure,
and stop after two consecutive rounds without a passing improvement. Python executes the recorded
experiment deterministically; hypothesis selection remains with the agent.

Every claim must cite the returned run ID plus its run card, checkpoint hash, data manifest hash,
suite hash, score ID, and event sequence. Never claim GPU validation without actual target-hardware
commands and their exit codes.
