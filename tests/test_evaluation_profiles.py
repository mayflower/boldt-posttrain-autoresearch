import json
from pathlib import Path

import pytest

from boldt_posttrain.evaluation import load_suite, register_promotion_suite


def test_profiles_and_external_promotion_registration(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "promotion.json"
    outside.write_text(
        json.dumps(
            {"cases": [{"case_id": "x", "category": "instruction", "prompt": "p", "expected": "a"}]}
        )
    )
    registered = register_promotion_suite(outside, repo / "registry.json", repo_root=repo)
    assert len(registered["suite_hash"]) == 64
    with pytest.raises(ValueError):
        register_promotion_suite(repo / "registry.json", repo / "other.json", repo_root=repo)
    assert load_suite(outside, profile="dev")["profile"] == "dev"


def test_proxy_is_a_strict_deterministic_subset_of_dev():
    suite = Path(__file__).resolve().parents[1] / "data/eval/dev.json"
    dev = load_suite(suite, profile="dev")["cases"]
    first = load_suite(suite, profile="proxy")["cases"]
    second = load_suite(suite, profile="proxy")["cases"]
    assert 0 < len(first) < len(dev)
    assert [case["case_id"] for case in first] == [case["case_id"] for case in second]
