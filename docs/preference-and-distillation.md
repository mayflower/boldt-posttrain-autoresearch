# Preference and distillation

DPO, KTO, and ORPO are separate real TRL paths. KTO explicitly expands each pair to balanced desirable/undesirable rows; ORPO uses the pinned `trl.experimental.orpo` implementation. Every path validates non-empty distinct answers and hard token/length-ratio limits before loading a trainer.

Offline distillation uses an exact local or revision-pinned teacher, stores teacher generations as an immutable data artifact, reruns language, deduplication, and leakage gates, then invokes the same SFT implementation used by normal training.
