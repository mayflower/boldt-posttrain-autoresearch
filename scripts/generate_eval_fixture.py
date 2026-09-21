#!/usr/bin/env python3
"""Regenerate the deterministic local development proxy from the real suite.

The proxy is the fast, small evaluation the search loop runs between full
evaluations. v1 was a separate hand-written fixture of eight templated prompts
("Antworte exakt mit Ja", "20 plus 22", a self-answering long context, a
"coding" case that only asked for the string "1 + 1"). It measured nothing the
full suite measures.

This regenerates dev.json as a deterministic subset of german-core-v2 -- the same
authored German tasks -- restricted to the categories the development scorer
(evaluation._default_validate) can judge faithfully: exact answers and exact
numeric results. The behavioural categories (safety, over-refusal, language,
JSON format) need the full validator dispatch in secure_compat.evaluation and are
covered by the full suite, not by this proxy.

Because it is a strict subset of the real suite, improving the proxy score and
improving the real score are the same direction, not two unrelated targets.

Run: uv run --locked python scripts/generate_eval_fixture.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The proxy maps each source category to the dev-scorer category name and how
# many cases to take. Deterministic: the first N cases of each family.
# Long context is deliberately excluded: its 9k-word prompts are the slowest to
# generate and defeat the point of a fast proxy. The full suite covers it.
SELECTION = [
    ("german_instruction", "instruction", 12),
    ("reasoning_core", "reasoning", 12),
]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_eval_suite", ROOT / "scripts/build_eval_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _expected(validator: dict) -> str:
    kind, params = validator["type"], validator["parameters"]
    if kind == "exact":
        return str(params["expected"])
    if kind == "numeric":
        value = params["expected"]
        return str(int(value)) if float(value).is_integer() else str(value)
    raise ValueError(f"proxy only carries exact/numeric cases, not {kind}")


def build_cases() -> list[dict]:
    builder = _load_builder()
    by_category: dict[str, list[dict]] = {}
    for source_case in builder.build():
        by_category.setdefault(source_case["category"], []).append(source_case)

    cases: list[dict] = []
    for source_category, dev_category, count in SELECTION:
        chosen = by_category[source_category][:count]
        for source_case in chosen:
            cases.append(
                {
                    "case_id": source_case["case_id"],
                    "category": dev_category,
                    "prompt": source_case["prompt"],
                    "expected": _expected(source_case["validator"]),
                }
            )
    return cases


def main() -> None:
    path = ROOT / "data/eval/dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "revision": "local-v2", "cases": build_cases()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
