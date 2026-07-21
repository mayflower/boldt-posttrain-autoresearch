import json
import time
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from boldt_posttrain.artifacts import ArtifactRef, sha256_file
from boldt_posttrain.merge import MergeError, execute_merge
from boldt_posttrain.merge import MergeInput, materialize_adapter
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import ResolvedModelRef
from boldt_posttrain.training import train_adapter
from tests.artifact_chain import initialized_repository
from tests.test_training_preflight import experiment
from tests.tiny_model import build_tiny_model


def full_model_ref(path: Path, repository: Path) -> dict:
    return ArtifactRef.from_path(
        path,
        role="full_checkpoint",
        media_type="application/vnd.boldt.transformers-checkpoint",
        relative_to=repository,
    ).to_dict()


@pytest.mark.parametrize("method", ["linear", "slerp", "ties", "dare_ties"])
def test_real_tiny_mergekit_output_loads_and_forwards(method: str, tmp_path: Path):
    repository = initialized_repository(tmp_path / "repo")
    first = build_tiny_model(repository / "models/first")
    second = build_tiny_model(repository / "models/second")
    second_model = AutoModelForCausalLM.from_pretrained(second)
    for parameter in second_model.parameters():
        parameter.data.add_(0.001)
    second_model.save_pretrained(second)
    tokenizer_hash = sha256_file(first / "tokenizer.json")
    policy = load_policy()
    metadata = {
        "kind": "local_full_checkpoint",
        "requested": "fixture",
        "base_model": {
            "repo_id": policy.seed_model["repo_id"],
            "revision": policy.seed_model["revision"],
        },
        "artifact": None,
        "tokenizer_sha256": tokenizer_hash,
        "chat_template_sha256": sha256_file(first / "chat_template.jinja"),
        "model_config_sha256": sha256_file(first / "config.json"),
        "architecture": "GPT2LMHeadModel",
        "source_run_id": None,
    }
    outputs = repository / "outputs/posttrain"
    result = execute_merge(
        method=method,
        model_paths=[first, second],
        input_refs=[full_model_ref(first, repository), full_model_ref(second, repository)],
        parent_run_ids=["fixture-one", "fixture-two"],
        model_metadata=metadata,
        data_metadata={
            "license_status": "usable",
            "leakage_statistics": {"status": "clean", "hit_count": 0},
        },
        parameters=(
            {"weights": [0.5, 0.5]}
            if method == "linear"
            else {"t": 0.5}
            if method == "slerp"
            else {"weight": 0.5, "density": 0.5}
        ),
        dtype="float32",
        output_checkpoint_root=outputs / "checkpoints",
        output_merge_root=outputs / "merge",
        state_root=outputs,
        policy=policy,
        allow_checkpoints=True,
        use_gpu=False,
        deadline=time.monotonic() + 120,
        repository_root=repository,
        base_model=str(first) if method in {"ties", "dare_ties"} else None,
    )
    checkpoint = Path(result["checkpoint"])
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model(**tokenizer("Hallo", return_tensors="pt"))
    card = json.loads((outputs / "runs" / result["run_id"] / "run_card.json").read_text())
    assert card["environment"]["mergekit_exit_code"] == 0
    assert (outputs / "merge" / result["run_id"] / "mergekit.yaml").is_file()


def test_checkpoint_permission_blocks_before_writes(tmp_path: Path):
    outputs = tmp_path / "outputs/posttrain"
    with pytest.raises(MergeError, match="permission"):
        execute_merge(
            method="linear",
            model_paths=[],
            input_refs=[],
            parent_run_ids=[],
            model_metadata={},
            data_metadata={},
            parameters={},
            dtype="float32",
            output_checkpoint_root=outputs / "checkpoints",
            output_merge_root=outputs / "merge",
            state_root=outputs,
            policy=load_policy(),
            allow_checkpoints=False,
            use_gpu=False,
            deadline=time.monotonic() + 1,
        )
    assert not (outputs / "checkpoints").exists()


def test_real_peft_adapter_materializes_to_full_checkpoint(tmp_path: Path, monkeypatch):
    from datasets import Dataset
    import boldt_posttrain.training as training_module

    repository = initialized_repository(tmp_path / "repo")
    outputs = repository / "outputs/posttrain"
    monkeypatch.setattr(training_module, "OUTPUTS", outputs)
    model_path = build_tiny_model(repository / "models/base")
    policy = load_policy()
    trained = train_adapter(
        kind="sft",
        model_source=str(model_path),
        revision=None,
        dataset=Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hallo"},
                        {"role": "assistant", "content": "Buch"},
                    ]
                }
            ]
        ),
        output_root=outputs / "checkpoints",
        policy=policy,
        experiment=experiment(),
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
        repository_root=repository,
        data_metadata={
            "license_status": "usable",
            "leakage_statistics": {"status": "clean", "hit_count": 0},
        },
    )
    card = json.loads((outputs / "runs" / trained["run_id"] / "run_card.json").read_text())
    checkpoint_ref = next(item for item in card["outputs"] if item["role"] == "adapter_checkpoint")
    model = card["model"]
    resolved = ResolvedModelRef(
        kind="peft_adapter",
        requested=trained["run_id"],
        base_model=model["base_model"],
        artifact=checkpoint_ref,
        tokenizer_sha256=model["tokenizer_sha256"],
        chat_template_sha256=model["chat_template_sha256"],
        model_config_sha256=model["model_config_sha256"],
        architecture=model["architecture"],
        source_run_id=trained["run_id"],
    )
    path, materialization_id = materialize_adapter(
        MergeInput(trained["run_id"], resolved, card, {"status": "passed"}),
        base_model_source=str(model_path),
        base_model_revision=None,
        output_root=outputs / "checkpoints",
        policy=policy,
        allow_checkpoints=True,
        repository_root=repository,
        state_root=outputs,
    )
    full = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    full(**tokenizer("Hallo", return_tensors="pt"))
    materialization_card = json.loads(
        (outputs / "runs" / materialization_id / "run_card.json").read_text()
    )
    assert materialization_card["parameters"]["operation"] == "peft_merge_and_unload"
