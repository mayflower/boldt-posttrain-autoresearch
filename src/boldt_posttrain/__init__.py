"""boldt_posttrain — shared, stdlib-only helpers for the German post-training AutoResearch loop.

The loop's core (config resolution, provenance/run cards, the deterministic scorer, the trial
recipe, and the frontier view) lives here so the ``scripts/pt_*.py`` CLIs stay thin and the
scoring/gate logic has ONE auditable definition. Heavy ML (torch/transformers/trl/peft/mergekit)
is imported lazily inside ``--real`` code paths only; importing this package pulls in nothing
beyond the standard library.
"""

__all__ = ["artifacts", "config", "policy", "provenance"]
__version__ = "0.1.0"
