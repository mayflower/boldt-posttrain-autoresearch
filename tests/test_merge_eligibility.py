from pathlib import Path

import pytest

from boldt_posttrain.merge import (
    MergeError,
    eligible_input,
    merge_configuration,
    merge_parameter_grid,
    validate_merge_parameters,
)
from boldt_posttrain.policy import load_policy
from boldt_posttrain.scoring import create_score
from tests.artifact_chain import complete_chain


def test_merge_accepts_only_verified_scored_real_candidate(tmp_path: Path, monkeypatch):
    chain = complete_chain(tmp_path, monkeypatch)
    create_score(
        chain["candidate_eval_run_id"],
        policy=chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    item = eligible_input(
        chain["candidate_run_id"],
        chain["policy"],
        outputs_root=chain["outputs"],
        repository_root=chain["repository"],
    )
    assert item.run_id == chain["candidate_run_id"]
    checkpoint = chain["outputs"] / "checkpoints" / chain["candidate_run_id"]
    (checkpoint / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(Exception, match="hash|candidate|verification"):
        eligible_input(
            chain["candidate_run_id"],
            chain["policy"],
            outputs_root=chain["outputs"],
            repository_root=chain["repository"],
        )


def test_mergekit_configs_use_exact_tokenizer_not_union(tmp_path: Path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    linear = merge_configuration(
        "linear",
        [first, second],
        dtype="float32",
        parameters={"weights": [0.4, 0.6]},
    )
    slerp = merge_configuration("slerp", [first, second], dtype="float32", parameters={"t": 0.5})
    ties = merge_configuration(
        "ties",
        [first, second],
        dtype="float32",
        parameters={"weight": 1.0, "density": 0.5},
        base_model="seed/model@" + "a" * 40,
    )
    assert "tokenizer_source: union" not in linear + slerp + ties
    assert "merge_method: linear" in linear
    assert "merge_method: slerp" in slerp
    assert "merge_method: ties" in ties


def test_empty_or_single_candidate_space_is_rejected(tmp_path: Path):
    with pytest.raises(MergeError, match="two or more"):
        merge_configuration("linear", [tmp_path / "one"], dtype="float32", parameters={})


def test_merge_parameter_grid_is_explicit_and_policy_bounded():
    grid = merge_parameter_grid(
        ["linear", "slerp"],
        {
            "linear": [{"weights": [0.5, 0.5]}, {"weights": [0.25, 0.75]}],
            "slerp": [{"t": 0.25}, {"t": 0.75}],
        },
    )
    assert len(grid) == 4
    bounds = load_policy().document["merge"]["parameter_bounds"]
    for method, parameters in grid:
        validate_merge_parameters(method, parameters, model_count=2, bounds=bounds)
    with pytest.raises(MergeError, match="sum to one"):
        validate_merge_parameters("linear", {"weights": [0.8, 0.8]}, model_count=2, bounds=bounds)
    with pytest.raises(MergeError, match="outside policy"):
        validate_merge_parameters("slerp", {"t": 2.0}, model_count=2, bounds=bounds)
