# Data pipeline

Discovery selects only training-usable sources from the protected organization/license allowlist.
Ordering is German sample ratio, known row count, then dataset/config/split. Limits are two SFT,
one preference, and one CPT source.

Preparation streams rows into schema/source-group shards bounded at 50,000 rows and 128 MiB.
SHA-256 content IDs provide exact deduplication; deterministic fixed-band MinHash buckets use a
temporary local SQLite file. Publication is an atomic directory swap and the SQLite file is removed
after success. Training reads streaming datasets and requires `max_steps`.

The decontamination corpus covers local prompts, contexts, expected answers, choices, documents,
all pinned lm-eval task documents/answers, and the registered promotion holdout. Manifests record
the exact corpus and policy hashes; stale data is rejected.

Failure mining emits only fixed categories/statistics. Synthesis never receives visible evaluation
prompts. Numeric, JSON-schema, extraction, ordered-term, language, and harmless non-refusal tasks
use mechanical verification. Best-of-N is capped at four and retains all candidate hashes.
