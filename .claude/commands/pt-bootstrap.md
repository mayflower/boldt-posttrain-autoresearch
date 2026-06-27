---
description: Create missing script/config stubs that satisfy the post-training AutoResearch contract
argument-hint: ""
allowed-tools: Bash(mkdir *) Bash(test *) Bash(chmod *) Bash(python -m py_compile *) Read Edit Write Glob Grep
disable-model-invocation: true
---
# PostTrain AutoResearch — bootstrap scripts

Create minimal, reviewable script stubs only if missing. Do not overwrite existing scripts.

Use `docs/posttrain-script-contracts.md` as the contract. Create stdlib-only stubs for:

- `scripts/pt_status.py`
- `scripts/pt_report.py`
- `scripts/pt_frontier_status.py`
- `scripts/check_posttrain_integrity.py`
- `scripts/pt_discover_openeurollm_de.py`
- `scripts/pt_prepare_openeurollm_de.py`
- `scripts/pt_baseline.py`
- `scripts/pt_train_specialist.py`
- `scripts/pt_train_preference.py`
- `scripts/pt_train_cpt.py`
- `scripts/pt_merge_search.py`
- `scripts/pt_eval.py`
- `scripts/pt_score.py`
- `scripts/pt_promote.py`
- `scripts/pt_log_result.py`

Stub behavior:

- parse common args;
- write JSON with `status`, `mode`, `message`, `missing_real_implementation` as appropriate;
- never fabricate model metrics;
- make dry-run paths validate configs and print a plan;
- real mode exits nonzero with a clear message until implemented.

After creating stubs:

```bash
python -m py_compile scripts/pt_*.py scripts/check_posttrain_integrity.py
python scripts/pt_status.py --format markdown
```

Report files created and the next command, usually `/pt-data dry`.
