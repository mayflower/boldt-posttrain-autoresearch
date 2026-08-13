"""Fixed-taxonomy failure mining and leakage-safe mechanically verified synthesis."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data_pipeline import (
    FastTextLanguageIdentifier,
    canonical_json,
    content_id,
    sha256_bytes,
    verify_hashed_artifact,
)

CATEGORIES = {
    "instruction",
    "format",
    "reasoning",
    "longcontext",
    "language",
    "over_refusal",
    "safety",
    "coding",
}


def mine_failures(
    summary: Mapping[str, Any],
    raw_generations: Sequence[Mapping[str, Any]],
    *,
    dimension_weights: Mapping[str, float],
) -> Dict[str, Any]:
    if summary.get("status") not in {"ok", "failed"}:
        raise ValueError("evaluation summary has no usable result status")
    counts: Counter[str] = Counter()
    case_ids: Dict[str, List[str]] = defaultdict(list)
    validators: Counter[str] = Counter()
    lengths: Dict[str, List[int]] = defaultdict(list)
    technical: Counter[str] = Counter()
    for row in raw_generations:
        if row.get("technical_error"):
            technical[str(row["technical_error"])] += 1
            continue
        category = str(row.get("category", "instruction"))
        if category not in CATEGORIES:
            raise ValueError(f"unknown failure category in evaluation: {category}")
        lengths[category].append(len(str(row.get("output", ""))))
        if row.get("correct") is False:
            counts[category] += 1
            case_ids[category].append(str(row.get("case_id")))
            validators.update(str(value) for value in row.get("validator_errors", []))
        if row.get("over_refusal_rate") == 1.0:
            counts["over_refusal"] += 1
            case_ids["over_refusal"].append(str(row.get("case_id")))
    categories = {}
    for category in sorted(CATEGORIES):
        category_lengths = lengths.get(category, [])
        categories[category] = {
            "count": counts[category],
            "case_ids": sorted(set(case_ids[category])),
            "mean_response_length": (
                sum(category_lengths) / len(category_lengths) if category_lengths else 0.0
            ),
            "priority": counts[category] * float(dimension_weights.get(category, 1.0)),
        }
    body = {
        "schema_version": 1,
        "status": "ok",
        "eval_run_id": summary.get("run_id"),
        "categories": categories,
        "validator_errors": dict(sorted(validators.items())),
        "technical_errors": dict(sorted(technical.items())),
    }
    return {**body, "artifact_hash": sha256_bytes(canonical_json(body))}


def _numeric_task(rng: random.Random) -> Dict[str, Any]:
    left, right = rng.randint(10, 999), rng.randint(2, 99)
    operation = rng.choice(["+", "-", "*"])
    answer = {"+": left + right, "-": left - right, "*": left * right}[operation]
    return {
        "task_type": "numeric",
        "prompt": f"Berechne {left} {operation} {right}.",
        "ground_truth": {"value": answer},
    }


def _json_task(rng: random.Random) -> Dict[str, Any]:
    name = f"Eintrag-{rng.randint(100, 999)}"
    count = rng.randint(1, 20)
    schema = {
        "type": "object",
        "required": ["name", "count"],
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "additionalProperties": False,
    }
    return {
        "task_type": "json_schema",
        "prompt": f'Gib ein JSON-Objekt mit name="{name}" und count={count} aus.',
        "ground_truth": {"schema": schema, "value": {"name": name, "count": count}},
    }


def _ordered_task(rng: random.Random) -> Dict[str, Any]:
    terms = rng.sample(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], 3)
    return {
        "task_type": "ordered_terms",
        "prompt": "Nenne diese Begriffe in der angegebenen Reihenfolge: " + ", ".join(terms),
        "ground_truth": {"terms": terms},
    }


def build_synthesis_tasks(
    failure_artifact: Mapping[str, Any],
    source_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 17,
    maximum: int = 128,
) -> List[Dict[str, Any]]:
    """Use only category statistics; evaluation prompts/answers are not accepted as input."""
    if not verify_hashed_artifact(failure_artifact):
        raise ValueError("failure artifact hash is invalid")
    rng = random.Random(seed)
    priorities = sorted(
        failure_artifact["categories"].items(), key=lambda item: (-item[1]["priority"], item[0])
    )
    active = [name for name, stats in priorities if stats["count"] > 0] or ["reasoning"]
    sources = iter(source_rows)
    tasks = []
    for index in range(maximum):
        category = active[index % len(active)]
        if category == "reasoning":
            task = _numeric_task(rng)
        elif category == "format":
            task = _json_task(rng)
        elif category in {"instruction", "coding"}:
            task = _ordered_task(rng)
        elif category in {"language", "over_refusal"}:
            task = {
                "task_type": "non_refusal" if category == "over_refusal" else "language",
                "prompt": "Antworte auf Deutsch mit den Begriffen Sonne und Energie.",
                "ground_truth": {"required_terms": ["Sonne", "Energie"]},
            }
        else:
            try:
                source = next(sources)
            except StopIteration:
                break
            text = str(source.get("text", source.get("content", "")))
            match = re.search(r"\[\[(.+?)\]\]", text)
            if not match:
                continue
            answer = match.group(1)
            task = {
                "task_type": "extraction",
                "prompt": f"Extrahiere den markierten Inhalt: {text}",
                "ground_truth": {"span": answer},
                "source_content_id": content_id(text),
                "license": source.get("license"),
            }
        task.setdefault("source_content_id", f"procedural:{seed}:{index}")
        tasks.append(task)
    return tasks


def verify_candidate(
    task: Mapping[str, Any],
    candidate: str,
    *,
    language_id: Optional[FastTextLanguageIdentifier] = None,
) -> Tuple[bool, str]:
    ground = task.get("ground_truth", {})
    kind = task.get("task_type")
    if kind == "numeric":
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", candidate)
        return (
            match is not None and float(match.group().replace(",", ".")) == ground["value"],
            "numeric_mismatch",
        )
    if kind == "json_schema":
        try:
            import jsonschema

            value = json.loads(candidate)
            jsonschema.validate(value, ground["schema"])
            return value == ground["value"], "json_value_mismatch"
        except (json.JSONDecodeError, jsonschema.ValidationError):
            return False, "invalid_json_schema"
    if kind == "extraction":
        return candidate.strip() == ground["span"], "span_mismatch"
    if kind == "ordered_terms":
        positions = [candidate.find(term) for term in ground["terms"]]
        return all(position >= 0 for position in positions) and positions == sorted(positions), (
            "ordered_terms_mismatch"
        )
    if kind in {"language", "non_refusal"}:
        terms_ok = all(term.casefold() in candidate.casefold() for term in ground["required_terms"])
        if kind == "language":
            if language_id is None:
                raise ValueError("language task verification requires the hashed language ID")
            language, _confidence = language_id.predict(candidate)
            return terms_ok and language == "de", "language_or_terms_mismatch"
        from .evaluation import is_refusal

        return terms_ok and not is_refusal(candidate), "refusal_or_terms_mismatch"
    raise ValueError(f"unsupported synthetic task type: {kind}")


def synthesize_verified(
    tasks: Sequence[Mapping[str, Any]],
    teacher: Callable[[Mapping[str, Any], int], str],
    *,
    teacher_ref: str,
    best_of_n: int,
    sampling: Mapping[str, Any],
    language_id: Optional[FastTextLanguageIdentifier] = None,
    eval_corpus: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not 1 <= best_of_n <= 4:
        raise ValueError("teacher best-of-N must be between one and four")
    sft, preferences, rlvr, discarded = [], [], [], 0
    eval_strings = [
        entry["canonical"].casefold() for entry in (eval_corpus or {}).get("entries", [])
    ]
    for task in tasks:
        prompt = str(task["prompt"])
        if any(value and value in prompt.casefold() for value in eval_strings):
            raise ValueError("synthetic task leaks a visible evaluation string")
        candidates = [teacher(task, index) for index in range(best_of_n)]
        verdicts = [verify_candidate(task, value, language_id=language_id) for value in candidates]
        valid = [index for index, verdict in enumerate(verdicts) if verdict[0]]
        if not valid:
            discarded += 1
            continue
        winner = min(valid, key=lambda index: (len(candidates[index]), index))
        metadata = {
            "task_type": task["task_type"],
            "source_content_id": task["source_content_id"],
            "teacher_ref": teacher_ref,
            "sampling": dict(sampling),
            "candidate_hashes": [
                hashlib.sha256(value.encode()).hexdigest() for value in candidates
            ],
            "teacher_candidates": candidates,
            "verifier_results": [{"valid": ok, "reason": reason} for ok, reason in verdicts],
            "license": task.get("license", "generated"),
            "leakage_gate": "clean",
            "language_gate": "checked" if task["task_type"] == "language" else "not_applicable",
        }
        sft.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "response": [{"role": "assistant", "content": candidates[winner]}],
                **metadata,
            }
        )
        rlvr.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "task_type": "exact" if task["task_type"] == "extraction" else task["task_type"],
                "ground_truth": (
                    {"value": task["ground_truth"]["span"]}
                    if task["task_type"] == "extraction"
                    else task["ground_truth"]
                ),
                "reward_version": 1,
                "source": {"content_id": task["source_content_id"], "teacher_ref": teacher_ref},
                "license": task.get("license", "generated"),
                "content_id": content_id(prompt),
                "leakage_clean": True,
                "training_usable": True,
            }
        )
        invalid = next((index for index, verdict in enumerate(verdicts) if not verdict[0]), None)
        if invalid is not None:
            preferences.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": [{"role": "assistant", "content": candidates[winner]}],
                    "rejected": [{"role": "assistant", "content": candidates[invalid]}],
                    **metadata,
                }
            )
    return {
        "status": "ok",
        "sft": sft,
        "preferences": preferences,
        "rlvr": rlvr,
        "discarded": discarded,
    }
