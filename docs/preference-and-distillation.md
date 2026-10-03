# Preference and online distillation

DPO, KTO and ORPO use separate pinned TRL trainers. KTO expands pairs to balanced
labels; ORPO uses `trl.experimental.orpo`. Prompt and answer lengths are validated
before trainer loading. `train preference --method` selects the recorded method.

OPD uses the pinned TRL 0.23.1 `GKDTrainer`, with `lmbda=1`, `seq_kd=False` and
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
