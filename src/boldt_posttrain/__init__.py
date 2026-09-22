"""boldt_posttrain — shared, stdlib-only helpers for the German post-training AutoResearch loop.

The loop's core (config resolution, provenance/run cards, the deterministic scorer, the trial
recipe, and the frontier view) lives here so the ``scripts/pt_*.py`` CLIs stay thin and the
scoring/gate logic has ONE auditable definition. Heavy ML (torch/transformers/trl/peft/mergekit)
is imported lazily inside ``--real`` code paths only; importing this package pulls in nothing
beyond the standard library.
"""

import importlib.util as _importlib_util
import os as _os

# Prefer HuggingFace's accelerated Rust download backend when it is installed
# (the `data` real extra ships it). setdefault keeps an operator override, and the
# find_spec guard avoids forcing it on a minimal install that lacks the package,
# where huggingface_hub would otherwise raise on the first download.
if _importlib_util.find_spec("hf_transfer") is not None:
    _os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

__all__ = [
    "bootstrap",
    "config",
    "data_pipeline",
    "distillation",
    "evaluation",
    "failure_mining",
    "frontier",
    "merge",
    "preference",
    "provenance",
    "recipe",
    "rewards",
    "rlvr",
    "verified_rl",
    "scheduler",
    "scoring",
    "training",
]
__version__ = "0.2.0"
