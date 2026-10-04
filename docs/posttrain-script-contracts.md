# Post-training CLI contracts

Use `uv run --locked python -m boldt_posttrain.cli`; it is the only entry point.

Canonical commands emit a JSON result and preserve nonzero statuses. Trainer and
third-party subprocess output may include progress logs; stdout is not guaranteed
to contain only one JSON object during real ML execution. Candidate-producing
commands require explicit `--dry-run` or `--real`; real training/merge additionally
require `--allow-gpu --allow-checkpoints`. Unsupported flags fail at parsing rather
than being silently ignored. Exit codes 0–5 are documented in `README.md`.

```bash
uv run --locked python -m boldt_posttrain.cli policy validate
uv run --locked python -m boldt_posttrain.cli integrity check --base-ref fb30e8228539d2dc76a9b4ce10813aa3f4268247
uv run --locked python -m boldt_posttrain.cli model resolve --candidate train-sft-20260721T120000.000000Z-0123456789abcdef
uv run --locked python -m boldt_posttrain.cli eval validate-suite
uv run --locked python -m boldt_posttrain.cli eval catalog
uv run --locked python -m boldt_posttrain.cli status
```

Canonical commands consume strict secure configs and publish schema-v1,
event-chained artifacts. Score accepts an exact evaluation run ID; promotion an
exact candidate run ID and base ref. Baseline, score, merge and promotion verify
their linked inputs. Run-card fields and roles live in `artifacts.py`.
