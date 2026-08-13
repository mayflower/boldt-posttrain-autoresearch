"""Fixed pure mechanical reward registry for verified RLOO data."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .data_pipeline import FastTextLanguageIdentifier
from .evaluation import is_refusal

REWARD_VERSION = 1


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    raise ValueError("completion must be text or a conversational message list")


def _applicable(task_type: str, *names: str) -> bool:
    return task_type in names


def exact_reward(
    completion: Any, *, task_type: str, ground_truth: Mapping[str, Any], **_kwargs: Any
) -> Optional[float]:
    if not _applicable(task_type, "exact"):
        return None
    expected = str(ground_truth.get("value", ground_truth.get("answer", ""))).strip()
    return float(completion_text(completion).strip() == expected)


def numeric_reward(
    completion: Any, *, task_type: str, ground_truth: Mapping[str, Any], **_kwargs: Any
) -> Optional[float]:
    if not _applicable(task_type, "numeric"):
        return None
    matches = re.findall(r"(?<!\w)[-+]?\d+(?:[.,]\d+)?", completion_text(completion))
    if not matches:
        return 0.0
    expected = float(ground_truth["value"])
    value = float(matches[-1].replace(",", "."))
    tolerance = float(ground_truth.get("tolerance", 0.0))
    return float(abs(value - expected) <= tolerance)


def json_schema_reward(
    completion: Any, *, task_type: str, ground_truth: Mapping[str, Any], **_kwargs: Any
) -> Optional[float]:
    if not _applicable(task_type, "json_schema"):
        return None
    try:
        import jsonschema

        value = json.loads(completion_text(completion))
        jsonschema.validate(value, ground_truth["schema"])
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return 0.0
    expected = ground_truth.get("value")
    return 1.0 if expected is None or value == expected else 0.0


def ordered_terms_reward(
    completion: Any, *, task_type: str, ground_truth: Mapping[str, Any], **_kwargs: Any
) -> Optional[float]:
    if not _applicable(task_type, "ordered_terms"):
        return None
    text = completion_text(completion).casefold()
    positions = [text.find(str(term).casefold()) for term in ground_truth["terms"]]
    return float(all(position >= 0 for position in positions) and positions == sorted(positions))


def german_language_reward(
    completion: Any,
    *,
    task_type: str,
    language_id: Optional[FastTextLanguageIdentifier] = None,
    ground_truth: Mapping[str, Any],
    **_kwargs: Any,
) -> Optional[float]:
    if not _applicable(task_type, "language"):
        return None
    if language_id is None:
        raise ValueError("german_language_reward requires the protected FastText identifier")
    text = completion_text(completion)
    language, confidence = language_id.predict(text)
    terms = ground_truth.get("required_terms", [])
    contains = all(str(term).casefold() in text.casefold() for term in terms)
    return float(language == "de" and confidence >= 0.8 and contains)


def non_refusal_reward(
    completion: Any, *, task_type: str, ground_truth: Mapping[str, Any], **_kwargs: Any
) -> Optional[float]:
    if not _applicable(task_type, "non_refusal"):
        return None
    text = completion_text(completion)
    terms = ground_truth.get("required_terms", [])
    contains = all(str(term).casefold() in text.casefold() for term in terms)
    return float(bool(text.strip()) and contains and not is_refusal(text))


def concise_length_reward(
    completion: Any, *, task_type: str, ground_truth: Mapping[str, Any], **_kwargs: Any
) -> Optional[float]:
    if task_type not in {
        "numeric",
        "json_schema",
        "exact",
        "ordered_terms",
        "language",
        "non_refusal",
    }:
        return None
    length = len(completion_text(completion).split())
    minimum = int(ground_truth.get("minimum_words", 1))
    maximum = int(ground_truth.get("maximum_words", 128))
    return float(minimum <= length <= maximum)


REGISTRY: Dict[str, Callable[..., Optional[float]]] = {
    "exact": exact_reward,
    "numeric": numeric_reward,
    "json_schema": json_schema_reward,
    "ordered_terms": ordered_terms_reward,
    "german_language": german_language_reward,
    "non_refusal": non_refusal_reward,
    "concise_length": concise_length_reward,
}
CORRECTNESS = {"exact", "numeric", "json_schema", "ordered_terms", "german_language", "non_refusal"}
BONUSES = {"german_language", "concise_length"}


def total_reward(
    completion: Any,
    *,
    task_type: str,
    ground_truth: Mapping[str, Any],
    weights: Mapping[str, float],
    clamp: Sequence[float],
    language_id: Optional[FastTextLanguageIdentifier] = None,
    log_component: Optional[Callable[[str, Optional[float], Optional[str]], None]] = None,
) -> float:
    """Evaluate the fixed registry, log details, enforce correctness, weight, and clamp."""
    if len(clamp) != 2 or not all(math.isfinite(float(value)) for value in clamp):
        raise ValueError("reward clamp must contain two finite values")
    parts: Dict[str, Optional[float]] = {}
    for name, function in REGISTRY.items():
        try:
            value = function(
                completion, task_type=task_type, ground_truth=ground_truth, language_id=language_id
            )
            if value is not None and not math.isfinite(value):
                raise ValueError("reward is NaN or infinite")
            parts[name] = value
            if log_component:
                log_component(name, value, None if value is not None else "not_applicable")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if log_component:
                log_component(name, None, f"{type(exc).__name__}: {exc}")
            raise
    applicable_correctness = [parts[name] for name in CORRECTNESS if parts[name] is not None]
    if not applicable_correctness:
        raise ValueError(f"no correctness reward applies to task type {task_type}")
    correctness = sum(applicable_correctness)
    score = 0.0
    for name, value in parts.items():
        if value is None:
            continue
        if name in BONUSES and correctness <= 0:
            continue
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight):
            raise ValueError("reward weight is NaN or infinite")
        score += weight * value
    return max(float(clamp[0]), min(float(clamp[1]), score))
