# Merge search

Merge inputs must be successful real candidate runs with hash-valid checkpoints, clean data,
usable licenses, exact seed compatibility, and passing real scores. PEFT adapters are materialized
with `merge_and_unload` into immutable full checkpoints before Mergekit runs.

The supported pinned Mergekit methods are `linear`, `slerp`, `ties`, and `dare_ties`. Every YAML,
checkpoint, exit code, and run card is retained. The tokenizer is always copied from an exact seed
descendant or the revision-pinned seed; union tokenizers and vocabulary expansion are forbidden.
