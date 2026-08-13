# Evaluation

`proxy` deterministically samples at least eight case IDs per local category and applies a small
lm-eval limit. `dev` runs the complete visible suite. Neither profile is promotable. `promotion`
reads only the externally registered suite whose exact hash must match.

Local generation classifies tokenization, context overflow, generation, out-of-memory, deadline,
and validator failures as technical. Any one makes the evaluation fail and is never counted as a
wrong model answer. Empty, incorrect, or refused responses remain model outcomes.

The same hashed FastText identifier used by preparation measures German retention after removing
URLs and code blocks. Desired safety refusal, harmless over-refusal, and overall refusal are
separate. Long-context cases are validated with the seed tokenizer and the protected token minimum.
