import copy
import json
import time
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM

from boldt_posttrain.artifacts import ArtifactRef, EventLog, atomic_write_json, sha256_file
from boldt_posttrain.evaluation import _publish_evaluation
from boldt_posttrain.merge import execute_merge
from boldt_posttrain.policy import load_policy
from boldt_posttrain.preference import train_preference_adapter
from boldt_posttrain.resolver import ResolvedModelRef, resolve_candidate
from boldt_posttrain.scoring import create_score, load_candidate_score
from boldt_posttrain.training import train_adapter
from tests.artifact_chain import initialized_repository
from tests.test_preference import preference_settings, rows as preference_rows
from tests.test_training_preflight import experiment
from tests.tiny_model import build_tiny_model


def tiny_policy(tmp_path: Path, model_path: Path):
    document = copy.deepcopy(load_policy().document)
    document["seed_model"].update(
        {
            "repo_id": str(model_path),
            "revision": "a" * 40,
            "architecture": "GPT2LMHeadModel",
            "model_type": "gpt2",
            "dtype": "float32",
            "context_length": 256,
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(model_path / "tokenizer_config.json"),
            "chat_template_sha256": sha256_file(model_path / "chat_template.jinja"),
            "special_tokens": {
                "bos_token": None,
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            },
        }
    )
    document["evaluation"]["bootstrap"]["samples"] = 50
    path = tmp_path / "tiny-policy.json"
    atomic_write_json(path, document)
    return load_policy(path)


def test_full_cpu_artifact_chain_rejects_checkpoint_mutation(tmp_path: Path, monkeypatch):
    import boldt_posttrain.evaluation as evaluation
    import boldt_posttrain.preference as preference_module
    import boldt_posttrain.training as training_module

    repository = initialized_repository(tmp_path / "repo")
    outputs = repository / "outputs/posttrain"
    model_path = build_tiny_model(repository / "fixtures/tiny-model")
    policy = tiny_policy(tmp_path, model_path)
    monkeypatch.setattr(training_module, "OUTPUTS", outputs)
    monkeypatch.setattr(preference_module, "OUTPUTS", outputs)

    fixture_data = repository / "fixtures/data"
    fixture_data.mkdir(parents=True)
    (fixture_data / "sft.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hallo"},
                    {"role": "assistant", "content": "Buch"},
                ]
            }
        )
        + "\n"
    )
    (fixture_data / "preference.jsonl").write_text(json.dumps(preference_rows()[0]) + "\n")
    (fixture_data / "cpt.jsonl").write_text(json.dumps({"text": "Regen fällt ."}) + "\n")
    data_metadata = {
        "status": "trainable",
        "license_status": "usable",
        "leakage_statistics": {"status": "clean", "hit_count": 0},
    }
    training = experiment()
    training["max_steps"] = 2
    candidate = train_adapter(
        kind="sft",
        model_source=str(model_path),
        revision=policy.seed_model["revision"],
        dataset=Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hallo"},
                        {"role": "assistant", "content": "Buch"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Regen"},
                        {"role": "assistant", "content": "Regen fällt"},
                    ]
                },
            ]
        ),
        output_root=outputs / "checkpoints",
        policy=policy,
        experiment=training,
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
        repository_root=repository,
        data_metadata=data_metadata,
    )
    resolved_candidate = resolve_candidate(candidate["run_id"], policy, outputs_root=outputs)
    baseline_model = ResolvedModelRef(
        kind="hub_model",
        requested=str(model_path),
        base_model={
            "repo_id": str(model_path),
            "revision": policy.seed_model["revision"],
        },
        artifact=None,
        tokenizer_sha256=policy.seed_model["tokenizer_sha256"],
        chat_template_sha256=policy.seed_model["chat_template_sha256"],
        model_config_sha256=policy.seed_model["model_config_sha256"],
        architecture=policy.seed_model["architecture"],
        source_run_id=None,
    )

    def deterministic_backend(resolved, cases, *, device):
        improved = resolved.kind == "peft_adapter"
        return [
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "output": "Buch",
                "score": 0.6 if improved and case["category"] == "german_instruction" else 0.5,
                "validator_detail": {
                    "empty": False,
                    "refusal": False,
                    "english_bleed": False,
                },
                "error": None,
            }
            for case in cases
        ]

    monkeypatch.setattr(evaluation, "generate_cases", deterministic_backend)
    monkeypatch.setattr(
        evaluation,
        "run_lm_eval",
        lambda *args, **kwargs: {
            task: 0.5 for task in policy.document["evaluation"]["lm_eval_tasks"]
        },
    )
    baseline = _publish_evaluation(
        resolved=baseline_model,
        policy=policy,
        config_path=evaluation.ROOT / "configs/posttrain/current.json",
        output_root=outputs / "baseline",
        baseline=True,
        replace_baseline=False,
        device="cpu",
        repository_root=repository,
    )
    evaluated = _publish_evaluation(
        resolved=resolved_candidate,
        policy=policy,
        config_path=evaluation.ROOT / "configs/posttrain/current.json",
        output_root=outputs / "evals",
        baseline=False,
        replace_baseline=False,
        device="cpu",
        repository_root=repository,
    )
    scored = create_score(
        evaluated["run_id"],
        policy=policy,
        outputs_root=outputs,
        repository_root=repository,
    )
    assert baseline["status"] == "succeeded"
    assert scored["status"] == "passed"
    EventLog(outputs).validate()
    checkpoint = Path(candidate["checkpoint"])
    weights = checkpoint / "adapter_model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original + b"tamper")
    with pytest.raises(Exception):
        load_candidate_score(
            candidate["run_id"],
            policy,
            outputs_root=outputs,
            repository_root=repository,
        )
    weights.write_bytes(original)

    preference_training = {**experiment(), "max_steps": 1, "per_device_batch_size": 1}
    preference = train_preference_adapter(
        method="dpo",
        model_source=str(model_path),
        revision=policy.seed_model["revision"],
        rows=preference_rows(),
        output_root=outputs / "checkpoints",
        policy=policy,
        training=preference_training,
        preference=preference_settings(),
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
        repository_root=repository,
        data_metadata=data_metadata,
    )
    assert preference["status"] == "succeeded"

    second = build_tiny_model(repository / "fixtures/tiny-model-two")
    second_model = AutoModelForCausalLM.from_pretrained(second)
    for parameter in second_model.parameters():
        parameter.data.add_(0.001)
    second_model.save_pretrained(second)
    full_refs = [
        ArtifactRef.from_path(
            path,
            role="full_checkpoint",
            media_type="application/vnd.boldt.transformers-checkpoint",
            relative_to=repository,
        ).to_dict()
        for path in (model_path, second)
    ]
    merged = execute_merge(
        method="linear",
        model_paths=[model_path, second],
        input_refs=full_refs,
        parent_run_ids=[candidate["run_id"], preference["run_id"]],
        model_metadata=baseline_model.to_dict(),
        data_metadata=data_metadata,
        parameters={"weights": [0.5, 0.5]},
        dtype="float32",
        output_checkpoint_root=outputs / "checkpoints",
        output_merge_root=outputs / "merge",
        state_root=outputs,
        policy=policy,
        allow_checkpoints=True,
        use_gpu=False,
        deadline=time.monotonic() + 120,
        repository_root=repository,
    )
    AutoModelForCausalLM.from_pretrained(merged["checkpoint"])
    EventLog(outputs).validate()
