# AUTORESEARCH — Boldt German Post-Training Loop

Operating manual for Claude Code / agentic runs. Read before changing anything.

## Goal

Post-train `mayflowergmbh/boldt-dc-1b-german-it-16k-dpo` into a stronger German-first small instruction model using auditable German portions of `openeurollm/*` datasets. The loop follows a Liquid-AI-style recipe adapted to a Llama-family 1B model: branch from one warm start, train small complementary specialists, merge promising checkpoints, evaluate, and promote only measured improvements.

## Default run modes

- `dry`: metadata, config, data-manifest, command-plan, schema, and integrity checks only. No GPU and no quality claims.
- `real`: may run data materialization, LoRA/QLoRA training, mergekit merge search, and evaluations. Requires explicit `--real` flags in scripts.
- Default loop budget is conservative. Commands pass budgets to scripts; scripts must enforce deadlines and write partial run cards instead of fabricating metrics.

## Base model assumptions

- Seed model: `mayflowergmbh/boldt-dc-1b-german-it-16k-dpo`.
- Keep tokenizer and chat template from the seed model.
- Preserve EOS handling from model config/generation config.
- New checkpoints must be descendants of the same architecture/tokenizer unless the run card marks them as non-mergeable.

## Data policy

The only default remote data source family is Hugging Face org `openeurollm`. Data discovery must be dynamic, not a hard-coded list.

A row/config/split is considered German candidate data if at least one holds:

- config/split/dataset name contains `de`, `deu`, `ger`, `german`, `deutsch`, or a known German collection name such as `german-commons`;
- a language-like column (`language`, `lang`, `locale`, `target_lang`, `source_lang`, etc.) is one of `de`, `deu`, `de-DE`, `German`, `Deutsch`;
- a sampled text/conversation passes a deterministic German-language classifier or a documented fastText/langid check.

The preparation step must normalize rows into one of these schemas:

```json
{"type":"sft", "messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}], "source":"...", "license":"..."}
{"type":"preference", "prompt":"...", "chosen":"...", "rejected":"...", "source":"...", "license":"..."}
{"type":"cpt", "text":"...", "source":"...", "license":"..."}
```

CPT rows are not mixed directly into SFT. They are only for tiny low-LR continued-pretraining/domain-refresh specialists and must be capped aggressively to avoid style drift.

Every prepared shard writes:

- `manifest.json`: dataset id, config, split, revision, row counts before/after filters, license/provenance, schema, checksum
- `leakage_report.json`: eval-set overlap / benchmark contamination status
- `quality_report.json`: German-language confidence, dedup stats, length distribution, refusal/safety flags

Unknown license, unknown language, missing leakage report, or missing manifest means the source is not trainable.

## Loop levers

The orchestrator chooses one lever per round based on measured state. Do not run in a fixed order.

| Lever | Purpose | Typical script |
|---|---|---|
| `data` | discover/materialize better German OpenEuroLLM subsets | `scripts/pt_prepare_openeurollm_de.py` |
| `sft-specialist` | train one small LoRA/QLoRA specialist | `scripts/pt_train_specialist.py` |
| `pref-specialist` | DPO/ORPO/KTO-style specialist from preference rows | `scripts/pt_train_preference.py` |
| `cpt-specialist` | tiny low-LR German domain refresh from high-quality raw text | `scripts/pt_train_cpt.py` |
| `merge` | merge complementary specialists / base / previous winner | `scripts/pt_merge_search.py` |
| `distill` | generate and train compact teacher-judged corrections | `scripts/pt_distill_trial.py` |
| `eval` | evaluate candidate and update frontier | `scripts/pt_eval.py`, `scripts/pt_score.py` |
| `promote` | copy only metadata pointers, not weights, into current best | `scripts/pt_promote.py` |

## Specialist taxonomy

Train complementary branches from a common warm start:

- `general-de`: German instruction following, concise helpfulness, style stability
- `reasoning-de`: math/logical/coding reasoning; prefer concise visible reasoning, avoid long `<think>` exposure unless intentionally training a reasoning variant
- `coding-de`: German explanations for code plus code correctness tasks
- `safety-de`: robust German refusal and safe redirection without over-refusal
- `longcontext-de`: 8k–16k summarization and multi-document instruction following
- `preference-de`: DPO/ORPO from preference rows or generated on-policy contrasts
- `raw-quality-de`: tiny CPT/domain refresh from high-quality German raw text only

## Merge policy

Merges are attempted only for same architecture/tokenizer descendants or fully materialized LoRA merges.

Candidate methods:

- linear/model-soup over close checkpoints
- SLERP for two related checkpoints
- TIES / DARE-TIES / DELLA when specialists have complementary deltas
- layer-wise interpolation only if all inputs share identical layer structure

Every merge search writes a merge matrix:

```json
{
  "candidate":"...",
  "parents":["..."],
  "method":"slerp|linear|ties|dare_ties|della",
  "parameters":{},
  "eval_summary":"outputs/posttrain/evals/<label>/summary.json",
  "verdict":"keep|reject|needs_eval"
}
```

## Evaluation gate

A candidate is promotable only if all required artifacts exist and pass:

- German instruction eval improves aggregate score over baseline/current winner.
- No material regression on German lm-eval core tasks beyond configured tolerance.
- No material regression in format-following / IFEval-style constraints.
- German language retention: low English bleed-through on German prompts.
- No response suppression: answer length, refusal rate, and empty-output rate stay within bounds.
- Safety set does not regress; over-refusal does not spike.
- Leakage status is verified clean.
- License/provenance is usable for the intended release.

Default German-core eval suggestions:

- `m_mmlu_de`, `arc_de`, `hellaswag_de`, `truthfulqa_de_mc2`, `belebele_deu_Latn`
- German instruction/style smoke set under `data/eval/german_chat_smoke.jsonl`
- ArenaHard-EU German rows, if available and kept strictly eval-only
- custom format-following cases in German
- long-context German summarization smoke tests

## Score sketch

Scripts may change weights, but the scorer must be deterministic and committed before a run:

```text
score =
  + 2.0 * Δgerman_instruction
  + 1.0 * Δformat_following
  + 1.0 * Δreasoning_core
  + 0.5 * Δlongcontext
  - 3.0 * max_lm_eval_regression_penalty
  - 2.0 * english_bleed_penalty
  - 2.0 * response_suppression_penalty
  - 3.0 * safety_regression_penalty
  - infinite if leakage/license/integrity fails
```

Dry-run rows can never pass promotion.

## Outer loop

`/pt-run` is the only autonomous loop. All other commands are single-step helpers.

One round:

1. Read state from artifacts: current config, frontier status, latest evals, latest run cards.
2. Pick one lever from the menu above.
3. Make one small config change if needed.
4. Run the chosen tool in dry or real mode.
5. Evaluate produced candidate if any.
6. Score and decide keep/reject.
7. Run integrity.
8. Stop early if promotable, if two consecutive rounds do not improve the frontier, or if integrity fails.

State lives on disk; there is no hidden controller and no private state file.

## Required artifacts

- `outputs/posttrain/data/manifest.json`
- `outputs/posttrain/baseline/summary.json`
- `outputs/posttrain/runs/<run_id>/run_card.json`
- `outputs/posttrain/evals/<label>/summary.json`
- `outputs/posttrain/frontier.json`
- `outputs/posttrain/results.tsv`

## Command table

| Command | What it does |
|---|---|
| `/pt-orient` | read rules, inspect repo/scripts, summarize current state |
| `/pt-status` | read-only artifact summary |
| `/pt-data [dry|real]` | discover and prepare German OpenEuroLLM data manifest/shards |
| `/pt-baseline [dry|real]` | establish baseline eval for the seed model |
| `/pt-trial [dry|real] [lever]` | run one chosen trial |
| `/pt-train [dry|real] <specialist>` | train one specialist branch |
| `/pt-merge [dry|real]` | run merge search over eligible candidates |
| `/pt-eval [dry|real] <model-or-label>` | evaluate one model/candidate |
| `/pt-promote <candidate>` | check promotion gate |
| `/pt-integrity [--base-ref REF]` | protected-surface check |
| `/pt-run [rounds] [dry|real]` | autonomous loop |
| `/pt-bootstrap` | create missing script/config stubs that match this contract |

## Hygiene

- Same seed, same eval prompt templates, same decoding settings for comparable evals.
- Saved model candidates under `outputs/posttrain/checkpoints/` are gitignored.
- Merge only specialists that share the warm-start basin.
- Keep all evals deterministic where possible: `temperature=0`, fixed max tokens, fixed seeds.
- When using synthetic/generated data, record teacher model, prompt, sampling params, filtering, and license implications.
