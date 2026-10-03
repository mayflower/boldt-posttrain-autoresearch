# Preference and online distillation

DPO, KTO and ORPO use separate pinned TRL trainers. KTO expands pairs to balanced
labels; ORPO uses `trl.experimental.orpo`. Prompt and answer lengths are validated
before trainer loading. `train preference --method` selects the recorded method.
Every trainer bounds sequences by `max_prompt_length + max_completion_length` (capped by
`training.context_length`), the same limits the length gates enforce, so accepted rows are
never cut by TRL's 1024-token default. `preference.rpo_alpha` adds RPO's chosen-answer NLL
term to DPO (TRL loss `"sft"` with weight `rpo_alpha`); it must be 0 for KTO and ORPO.

OPD uses the pinned TRL 1.14.1 `trl.experimental.gkd.GKDTrainer`, with `lmbda=1`, `seq_kd=False` and
`beta=1`: the current student generates fresh completions during training and
learns from the frozen teacher distribution on those completions. The subclass
records rollout tokens, step, content ID, log probabilities and token KL in the
hashed run artifact. `distill` and `train distill` are aliases for OPD. No offline
teacher-completion dataset feeds this path. The exact initial student checkpoint
is rejected as an OPD teacher; select a distinct licensed teacher revision.

SDPO conditions the self-teacher on feedback or a verified demonstration. SDFT
conditions it on demonstrations and updates an EMA teacher. These are custom
feedback-conditioned trainers on the locked stack. Newer TRL releases offer
experimental SDPO/SDFT trainers, but adopting them requires a separate dependency
migration and contract checks; they are not available in pinned TRL 0.23.1.

All three distillation methods use current student rollouts and completion-token
distribution matching. They publish canonical checkpoints and proceed through the
same resolver, evaluation, score and optional promotion gates as SFT.

GRPO and RLVR use native `GRPOTrainer` and `RLOOTrainer`, respectively. Rewards come
from verified training rows, never the protected evaluation corpus. Prepared data
must contain the appropriate verified tasks, feedback or demonstrations; changing
only the lever on the default SFT experiment does not supply those prerequisites.

## Sequence-level distillation (`seqkd`)

`opd`/`distill` compare teacher and student logits token by token, so the teacher must use the
student's exact tokenizer and chat template. Boldt only exists at 1B, so no stronger teacher
qualifies. `seqkd` moves knowledge as text instead and has two stages:

1. `pt seqkd generate --real --allow-gpu --config <experiment>` (outside the loop budget):
   - The experiment's `seqkd.teacher` must match an exact `repo_id@commit` entry in
     `policy.teachers` (human-owned) with `purpose: "seqkd"`; the model-card license must equal
     the entry's license.
   - Prompts are the conversation prefixes up to the first assistant turn of the verified SFT
     manifest, sampled deterministically (`max_prompts`, `seed`).
   - The teacher answers through offline vLLM with thinking disabled
     (`enable_thinking: false`) and the configured sampling. Answers that are empty, truncated,
     contain thinking markers, or whose prompt does not fit `training.context_length` are dropped.
   - Surviving rows pass the `data prepare` gates: German language ID, exact and near dedup,
     and the benchmark-leakage filter. Any leakage hit aborts the run.
   - The result is published under `outputs/posttrain/seqkd/<run_id>/` (shard, generation and
     leakage reports, manifest) with an event-anchored run card of type `seqkd_generate`.
2. The `seqkd` lever (`train seqkd` or `loop run`) verifies the run pinned in
   `seqkd.generation_run` (event chain, artifact hashes, current policy and eval-suite hash,
   teacher, unchanged generation parameters) and trains the student with the normal SFT/QLoRA
   path. Its run card has type `train_seqkd`, names the generation run as parent and records
   teacher and prompt source in its lineage.

The approved teacher is `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
(Apache-2.0). It needs transformers >= 5.8 and the `teacher` extra (vLLM). On Ampere GPUs
(RTX A6000) the FP8 checkpoint runs through vLLM's Marlin weight-only kernels.

