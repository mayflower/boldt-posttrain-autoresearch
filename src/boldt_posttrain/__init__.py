"""German post-training loop; heavy ML dependencies load only during execution."""

from importlib.metadata import version as _version

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
