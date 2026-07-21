# Scoring and promotion

`pt score --candidate <eval-run-id>` accepts only a schema-v1 real evaluation whose summary,
raw generations, lm-eval output, resolved model, source checkpoint, run card, policy hash, suite
hash, and event-chain anchor all verify. It computes deterministic paired bootstrap intervals and
writes a hash-referenced score run. A rejected score exits with code 1.

`pt promote --candidate <training-run-id> --base-ref <commit>` reloads and recomputes that score,
runs default-deny Git integrity, requires every protected gate, and updates
`outputs/posttrain/frontier/current.json` under an exclusive compare-and-swap lock. Promotion
history is immutable and no weights are moved.
