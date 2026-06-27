# Post-Training Script Contracts

The `.claude` commands call scripts by contract. Implement them in your repo or adapt command names to your existing trainer.

Each script must write machine-readable JSON and never fabricate metrics. If prerequisites are missing, return a clear failure JSON.

## Common flags

All scripts should support:

```bash
--config configs/posttrain/current.json
--out outputs/posttrain/<area>
--format json|markdown
--dry-run
--real
--allow-gpu
```

Real training scripts should also support budget/deadline flags:

```bash
--budget-minutes <int>
--allow-checkpoints
```

## scripts/pt_discover_openeurollm_de.py

Purpose: inspect Hugging Face org `openeurollm`, dataset configs, split names, schemas, and sample rows. Output candidate German sources.

Required output:

```json
{
  "status": "ok|fail",
  "org": "openeurollm",
  "candidates": [
    {
      "dataset_id": "openeurollm/...",
      "config": "...",
      "split": "train",
      "reason": "language_column|config_name|sample_langid",
      "schema_guess": "sft|preference|cpt|unknown",
      "license": "...|unknown",
      "row_estimate": 0,
      "training_usable": false
    }
  ]
}
```

Implementation hint:

- Use `huggingface_hub.HfApi().list_datasets(author="openeurollm")`.
- Use `datasets.get_dataset_config_names` and `datasets.get_dataset_split_names`.
- Stream small samples with `load_dataset(..., streaming=True)`.
- Do not download full datasets during discovery.

## scripts/pt_prepare_openeurollm_de.py

Purpose: materialize trainable German shards only after license/language/leakage checks.

Required artifacts:

- `outputs/posttrain/data/manifest.json`
- `outputs/posttrain/data/train_sft.jsonl`
- `outputs/posttrain/data/train_preference.jsonl`
- `outputs/posttrain/data/train_cpt.jsonl`
- `outputs/posttrain/data/leakage_report.json`
- `outputs/posttrain/data/quality_report.json`

## scripts/pt_baseline.py

Purpose: create reproducible baseline eval for seed model.

Required output:

- `outputs/posttrain/baseline/summary.json`
- `outputs/posttrain/baseline/run_card.json`

## scripts/pt_train_specialist.py

Purpose: train LoRA/QLoRA SFT or CPT specialist.

Required output:

- checkpoint/adapters under `outputs/posttrain/checkpoints/<run_id>/`
- `outputs/posttrain/runs/<run_id>/run_card.json`

The run card must include: base model, data manifest path/checksum, training args, seed, git commit, hardware, wall clock, trainable parameters, and path to adapter/full checkpoint.

## scripts/pt_train_preference.py

Purpose: DPO/ORPO/KTO-like training on preference rows. It must check chosen/rejected lengths and avoid response suppression.

Required output: same run card contract as `pt_train_specialist.py`, plus preference stats.

## scripts/pt_merge_search.py

Purpose: merge eligible candidates using mergekit or local weight interpolation.

Required inputs:

- eligible checkpoints from `outputs/posttrain/runs/*/run_card.json`
- merge config from `configs/posttrain/current.json`

Required output:

- `outputs/posttrain/merge/<merge_id>/merge_matrix.json`
- one run card per merged candidate

## scripts/pt_eval.py

Purpose: evaluate seed, specialist, or merged model.

Required output:

```json
{
  "status": "ok|fail",
  "model": "...",
  "label": "...",
  "metrics": {
    "german_instruction": 0.0,
    "format_following": 0.0,
    "reasoning_core": 0.0,
    "longcontext": 0.0,
    "english_bleed_rate": 0.0,
    "empty_output_rate": 0.0,
    "refusal_rate": 0.0,
    "lm_eval": {}
  },
  "artifacts": {}
}
```

## scripts/pt_score.py

Purpose: deterministic scoring against baseline and current frontier. Dry runs can never pass.

## scripts/pt_promote.py

Purpose: promotion gate. It may write `outputs/posttrain/frontier.json` only when all gates pass. It must not move/commit weights.

## scripts/pt_frontier_status.py and scripts/pt_report.py

Purpose: read-only summaries for Claude Code.

## scripts/check_posttrain_integrity.py

Purpose: fail if the loop touched protected surfaces. It should compare against a supplied base ref or current git state.
