from collections import Counter
from pathlib import Path

from boldt_posttrain.evaluation import (
    CATEGORY_MINIMUMS,
    generate_cases,
    load_suite,
    score_output,
    suite_hash,
)
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import resolve_model
from tests.tiny_model import build_tiny_model


def test_german_core_suite_has_required_case_counts_and_8k_contexts():
    cases = load_suite()
    counts = Counter(item["category"] for item in cases)
    assert len(cases) == 294
    for category, minimum in CATEGORY_MINIMUMS.items():
        assert counts[category] >= minimum
    assert all(
        len(item["prompt"].split()) >= 8000 for item in cases if item["category"] == "longcontext"
    )
    assert len(suite_hash()) == 64


def test_mechanical_validators_are_fail_closed():
    exact = {"validator": {"type": "exact", "parameters": {"expected": "ja"}}}
    assert score_output(exact, "ja")[0] == 1
    assert score_output(exact, "ja extra")[0] == 0
    numeric = {"validator": {"type": "numeric", "parameters": {"expected": 42, "tolerance": 0}}}
    assert score_output(numeric, "42")[0] == 1
    assert score_output(numeric, "42 ungefähr")[0] == 0


def test_real_transformers_generation_path_on_cpu(tmp_path: Path):
    model_path = build_tiny_model(tmp_path / "tiny")
    resolved = resolve_model(policy=load_policy(), model=str(model_path), external_roots=[tmp_path])
    cases = [
        {
            "case_id": "one",
            "category": "german_instruction",
            "prompt": "Hallo",
            "validator": {"type": "regex", "parameters": {"pattern": ".*"}},
            "max_new_tokens": 2,
        }
    ]
    records = generate_cases(resolved, cases, device="cpu")
    assert len(records) == 1
    assert records[0]["error"] is None
    assert isinstance(records[0]["output"], str)
