# Data pipeline

Discovery enumerates protected Hugging Face organizations dynamically, resolves every dataset to a commit, samples by streaming, and records license, schema, language, privacy, and remote-code evidence.

Preparation accepts only exact commits and allowed SPDX licenses, uses the checked-in fastText model by hash, normalizes explicit SFT/preference/CPT schemas, and applies cross-source exact/MinHash deduplication plus eval-corpus leakage filtering. Real shards are staged and atomically published below `outputs/posttrain/data/<run-id>/`; `current.json` appears only after every gate passes. No dataset is uploaded.
