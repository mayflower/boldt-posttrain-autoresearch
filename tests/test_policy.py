import json
from pathlib import Path

import pytest

from boldt_posttrain.policy import PolicyError, load_policy


def test_repository_policy_is_valid_and_revision_is_fixed():
    policy = load_policy()
    assert len(policy.seed_model["revision"]) == 40
    assert policy.seed_model["revision"] not in {"main", "master", "latest"}


def test_policy_rejects_unknown_key(tmp_path: Path):
    policy = load_policy().to_dict()
    policy["unknown"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(PolicyError, match="unknown policy.unknown"):
        load_policy(path)


def test_policy_rejects_unknown_nested_key(tmp_path: Path):
    policy = load_policy().to_dict()
    policy["scoring"]["promotion"]["agent_override"] = 1
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(PolicyError, match="agent_override"):
        load_policy(path)


def test_policy_rejects_movable_revision(tmp_path: Path):
    policy = load_policy().to_dict()
    policy["seed_model"]["revision"] = "main"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(PolicyError, match="exact 40-character"):
        load_policy(path)


def test_policy_and_base_are_protected():
    protected = load_policy().integrity["protected_globs"]
    assert "configs/posttrain/policy.json" in protected
    assert "configs/posttrain/base.json" in protected
