# German-core evaluation

`german-core-v1` combines 294 protected mechanical cases with three revision-pinned lm-eval tasks. Generation is greedy with the seed chat template, fixed seeds, explicit EOS/pad IDs, and complete prompt/output logging. Empty outputs and generation exceptions remain scored failures.

Validate without loading a model:

```bash
python -m boldt_posttrain.cli eval validate-suite
python -m boldt_posttrain.cli eval catalog
```

Real baseline and candidate evaluation require explicit GPU permission. Baselines are immutable and `baseline/current.json` is replaced only with `--replace-baseline`.
