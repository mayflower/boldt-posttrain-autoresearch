"""German post-training loop; heavy ML dependencies load only during execution."""

import importlib.util as _importlib_util
import os as _os
from importlib.metadata import version as _version

# Compatibility with the locked huggingface-hub 0.36 download backend.
# Hub 1.x uses hf-xet instead; remove this together with the dependency migration.
# Prefer the accelerated Rust download backend when it is installed
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
__version__ = _version("boldt-posttrain-autoresearch")
