# Post-Training CLI Contracts

All `scripts/pt_*.py` files are thin compatibility entrypoints for
`python -m boldt_posttrain.cli`. Fachlogik exists only in `src/boldt_posttrain/`.

Every CLI emits exactly one JSON object on stdout, detailed logs on stderr, and one of exit codes
0–5 documented in `README.md`. Subprocesses use argument arrays without a shell. Mutating commands
require exactly one explicit mode; dry plans cannot write any real namespace.

## Commands

```bash
python -m boldt_posttrain.cli policy validate
python -m boldt_posttrain.cli integrity check --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247
python -m boldt_posttrain.cli model resolve --candidate train-sft-20260721T120000.000000Z-0123456789abcdef
python -m boldt_posttrain.cli eval validate-suite
python -m boldt_posttrain.cli eval catalog
python -m boldt_posttrain.cli status
python -m boldt_posttrain.cli report
```

Data runs publish immutable discovery or prepared-data directories. Training publishes a verified
PEFT adapter only after save/reload/forward validation. Evaluation publishes summary, raw
generations, exact resolved-model JSON, lm-eval output, and a run card. Score and promotion accept
only those linked artifacts. Merge materializes PEFT inputs and delegates all full-weight merges
to the pinned Mergekit CLI.

Run-card schema v1 fields and artifact roles are defined in `src/boldt_posttrain/artifacts.py`.
