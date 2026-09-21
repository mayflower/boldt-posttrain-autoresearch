"""One definition of "is this answer correct" for numeric tasks.

Three places used to decide this independently: the evaluation validator
required the whole trimmed output to be the number, the RL reward took the LAST
number anywhere in the text, and failure-mining synthesis took the FIRST. For
expected value 42 that made "20 + 22 = 42" rejected by synthesis but rewarded by
RL, and "42 ist falsch. Das Ergebnis ist 7." the other way round -- and synthesis
turns those verdicts into chosen/rejected preference pairs.

The evaluation rule is the canonical one, because it is what decides promotion:
training against a looser reward than the scored metric is the misalignment, not
a convenience. Picking one number out of an ambiguous answer is guesswork, so an
answer that is not exactly a number is simply not correct.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# Deliberately no comma decimals: this mirrors the evaluation validator, and the
# prompts that carry a numeric validator ask for the bare number.
NUMERIC_ANSWER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def numeric_answer(text: str) -> Optional[float]:
    """Return the answer iff the trimmed output is exactly one number."""
    stripped = str(text).strip()
    if not NUMERIC_ANSWER.fullmatch(stripped):
        return None
    return float(stripped)


def numeric_matches(text: str, expected: Any, tolerance: Any = 0.0) -> bool:
    value = numeric_answer(text)
    if value is None:
        return False
    return abs(value - float(expected)) <= float(tolerance)


def numeric_matches_ground_truth(text: str, ground_truth: Mapping[str, Any]) -> bool:
    return numeric_matches(text, ground_truth["value"], ground_truth.get("tolerance", 0.0))
