# Architecture and remaining consolidation

The canonical experiment schema is `secure_compat.config.load_experiment`, with
`configs/posttrain/secure-current.json` as default. Human-owned rules remain in
`configs/posttrain/policy.json`.

Data discover/prepare, baseline, train, eval, score, merge, loop, promote, status
and report now use the canonical artifact lifecycle. `runtime_cli` adapts manual
commands; `loop` orchestrates the same producers. Compatibility scripts delegate
to those CLI verbs and preserve exit codes. Prepared data uses `data/current.json`;
baseline and frontier use verified pointers; candidates use schema-v1 cards and
the hash-chained event log. Recipe manifests and labels cannot enter these gates.

GRPO/RLOO use native TRL trainers. OPD uses native GKDTrainer with a provenance
subclass. SDPO/SDFT remain custom on the pinned stack; newer experimental TRL
implementations need a tested dependency migration. The old offline distillation
producer has been removed.

Remaining legacy code is active in the recipe-only bootstrap, failure synthesis,
search, mix and comparison utilities, which still use `current.json`,
`recipe-policy.json`, top-level recipe helpers and their own artifact formats.
They are not canonical candidate producers and their outputs cannot be promoted
by the secure loop. Removing those APIs requires either replacing their useful
capabilities with canonical producers or explicitly retiring them; their modules
must not be deleted as if unused. The custom recipe MinHash index and weighted
interleaving are further replacement candidates; the secure data path already
uses datasketch. Renaming `secure_compat` is cosmetic until this dependency split
is resolved.

Tests cover the canonical CLI wiring, artifact-chain rejection and tiny-model
training/save/reload. CUDA fixture results do not establish full Boldt training
quality or production memory acceptance. Production acceptance still requires
exact run IDs, artifact hashes, score IDs and event evidence on target hardware.
