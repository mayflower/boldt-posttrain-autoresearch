# Training

SFT and CPT use the pinned TRL `SFTTrainer` directly. GPU QLoRA is fixed to NF4, double quantization, BF16 compute, explicit target modules, and no automatic batch or method fallback. CPT consumes only `text` shards and enforces its protected learning-rate cap.

Training requires `--real --allow-gpu --allow-checkpoints`. Adapters are written to staging, loaded for a forward smoke test, hash-verified, and atomically published. A monotonic deadline stops at a step boundary; a valid, reload-verified checkpoint is published with `status="succeeded"` and `stop_reason="budget_limit"` when that boundary is reached.
