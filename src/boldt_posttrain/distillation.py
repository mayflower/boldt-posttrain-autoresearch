"""Bounded offline distillation with deterministic CPT task conversion."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def cpt_tasks(text: str, source_content_id: str) -> List[Dict[str, Any]]:
    """Turn raw CPT prose into simple tasks; never expose it as an unframed user prompt."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = [
        sentence.strip() for sentence in _SENTENCE_RE.split(normalized) if sentence.strip()
    ]
    first = sentences[0]
    fact = first.rstrip(".!?")
    return [
        {
            "task_type": "summary",
            "source_content_id": source_content_id,
            "prompt": [
                {
                    "role": "user",
                    "content": f"Fasse den folgenden Text knapp zusammen:\n\n{normalized}",
                }
            ],
        },
        {
            "task_type": "fact_extraction",
            "source_content_id": source_content_id,
            "prompt": [
                {
                    "role": "user",
                    "content": f"Extrahiere eine ausdrücklich genannte Tatsache aus dem Text:\n\n{normalized}",
                }
            ],
            "ground_truth": fact,
        },
        {
            "task_type": "explicit_question",
            "source_content_id": source_content_id,
            "prompt": [
                {
                    "role": "user",
                    "content": f"Welche Information wird im ersten Satz ausdrücklich genannt?\n\n{normalized}",
                }
            ],
            "ground_truth": fact,
        },
    ]


def _expired(deadline: float) -> bool:
    return time.monotonic() >= deadline


def run_distillation(
    *,
    prompts: Iterable[Mapping[str, Any]],
    generate: Callable[[Mapping[str, Any]], str],
    filter_output: Callable[[Mapping[str, Any], str], bool],
    train_student: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    deadline: float,
    register_candidate: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute teacher generation and student training with all required deadline boundaries."""
    accepted: List[Dict[str, Any]] = []
    generated = 0
    for prompt in prompts:
        if _expired(deadline):  # before every prompt
            return {
                "status": "budget_exhausted",
                "generated": generated,
                "accepted": len(accepted),
                "candidate_registered": False,
            }
        output = generate(prompt)
        generated += 1
        if _expired(deadline):  # after every generation
            return {
                "status": "budget_exhausted",
                "generated": generated,
                "accepted": len(accepted),
                "candidate_registered": False,
            }
        if _expired(deadline):  # explicit pre-filter boundary
            return {
                "status": "budget_exhausted",
                "generated": generated,
                "accepted": len(accepted),
                "candidate_registered": False,
            }
        if filter_output(prompt, output):
            accepted.append(
                {
                    **dict(prompt),
                    "response": [{"role": "assistant", "content": output}],
                    "teacher_output_hash": hashlib.sha256(output.encode()).hexdigest(),
                }
            )
    if _expired(deadline):  # before student training
        return {
            "status": "budget_exhausted",
            "generated": generated,
            "accepted": len(accepted),
            "candidate_registered": False,
        }
    if not accepted:
        return {
            "status": "rejected",
            "generated": generated,
            "accepted": 0,
            "candidate_registered": False,
        }
    training = dict(train_student(accepted))
    if training.get("status") not in {"ok", "pass"}:
        return {
            "status": "failed",
            "generated": generated,
            "accepted": len(accepted),
            "candidate_registered": False,
            "training": training,
        }
    result = {
        "status": "ok",
        "generated": generated,
        "accepted": len(accepted),
        "candidate_registered": register_candidate is not None,
        "training": training,
    }
    if register_candidate is not None:
        register_candidate(result)
    return result


from .secure_compat.distillation import (  # noqa: E402, F401
    DistillationError,
    _teacher_license,
    distill_and_train,
    extract_prompts,
)
