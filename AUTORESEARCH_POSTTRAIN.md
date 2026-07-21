# AutoResearch Operating Policy

## Research Objective

Improve German instruction following while preserving format following, reasoning, long context,
German retention, safety, and calibrated refusal behavior. Every candidate descends from the exact
seed commit and retains its tokenizer, chat template, special tokens, and architecture.

## Serial Research

Rounds are serial. The agent records one hypothesis and one lever in a strict experiment JSON.
Candidate-producing levers are SFT/QLoRA, CPT, DPO, KTO, ORPO, offline local distillation, and
Mergekit (`linear`, `slerp`, `ties`, `dare_ties`). Data discovery/preparation and baseline creation
are prerequisite single-step operations, not candidate rounds.

## Data

Discovery enumerates the allowed Hub organizations dynamically and records exact dataset commits,
configs, splits, licenses, schema evidence, language evidence, and sample hashes. Preparation uses
the checked-in hash-pinned fastText model, explicit schema adapters, exact SHA-256 deduplication,
MinHash near-deduplication, and exact/near leakage comparison against the protected full eval
corpus. Unknown license, language failure, stream interruption, stale suite hash, or leakage blocks
training. CPT is separate and policy-capped.

## Evaluation and Promotion

The 294-case `german-core-v1` suite and three revision-pinned local lm-eval tasks are mandatory.
Generation is greedy and seed-fixed. Empty outputs and exceptions are scored failures. Scoring
requires all policy metrics, aligned per-case evidence, finite rates, exact policy/suite hashes,
the candidate checkpoint bytes, raw generations, lm-eval output, run cards, and event anchors.
Paired bootstrap intervals use the protected seed and sample count.

Promotion requires a positive weighted score, the protected German-instruction gain, every hard
regression gate, clean data, usable licenses, a better score than the current frontier, intact Git
integrity, and an intact event chain. Promotion updates metadata only; it never moves weights.

## Budgets and Failures

A loop deadline is set once and propagated to training, evaluation, and merge. Training stops only
at a step boundary and marks `budget_exhausted`; evaluation checks case boundaries and times out
lm-eval; merge starts no new candidate after deadline. OOM and dependency failures terminate the
operation. No retry changes semantics.

## Artifacts

Runtime identity is a unique run ID, never a label. Relevant files and directories are
deterministically hashed. JSON and pointers are fsynced and atomically replaced; shared pointers
use exclusive locks. Runtime status comes only from verified schema-v1 artifacts and the hash
chain. Legacy files are displayed as unverified and never influence scoring, merge, or promotion.
