# Training

SFT, CPT, DPO, KTO, and ORPO use the single locked TRL/Transformers/PEFT set. Preference artifacts
remain conversational. Supported PEFT recipes are default LoRA, rsLoRA, and PiSSA
`pissa_niter_4`. Liger is opt-in for supported SFT/DPO/KTO measurements and is never enabled
automatically.

RLVR uses only `RLOOTrainer`, PEFT LoRA, four generations by default, 256 completion tokens, and no
rollout server. Exact, numeric, JSON-schema, ordered-term, German-language, harmless non-refusal,
and concise-length rewards are fixed pure mechanical functions. Invalid reward execution is a
technical failure. Reward sums use protected weights/clamps and bonuses cannot replace correctness.

Run cards include GPU seconds, tokens and throughput, peak VRAM, trainable parameters, checkpoint
bytes, and proxy score where applicable. Data-mix runs additionally record actual tokens by group
and explicit repeat counts.
