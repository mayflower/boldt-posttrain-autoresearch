# Post-training command contracts

The authoritative implementation is the `pt` CLI and the domain modules in
`src/boldt_posttrain`. Commands return `0` for success, `1` for a technically successful rejection,
`2` for invalid arguments/configuration, `3` for a missing prerequisite, `4` for a technical
execution failure, and `5` for an integrity failure.

All real GPU commands require explicit `--real --allow-gpu --device cuda:N`. Commands that create
checkpoints additionally require `--allow-checkpoints`. Merge placement is separately controlled by
`--merge-device cpu|cuda:N`. There is no implicit CPU or device fallback.

Every productive run writes a run card and immutable content hashes. Events are appended to the
existing JSONL hash chain. Evaluation summaries always include profile, suite/decontamination
hashes, technical/model error counts, and raw-generation artifact hashes.
