"""Online objectives, secure-loop routing, and real CUDA fixture candidates."""

import copy
import json
import time
from pathlib import Path

import pytest

from boldt_posttrain import cli, loop
from boldt_posttrain.artifacts import ArtifactRef, EventLog, sha256_file
from boldt_posttrain.online import (
    ONLINE_LEVERS,
    online_rows,
    online_settings,
    self_teacher_prompt,
    split_online_rows,
    validate_online_policy,
    verify_response,
)
from boldt_posttrain.online_training import reverse_kl, train_online_candidate, update_ema_teacher
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import ResolvedModelRef, resolve_candidate
from boldt_posttrain.config import ExperimentConfig, validate_config_dict
from boldt_posttrain.data_pipeline import DataError, normalize_row, row_texts
from tests.artifact_chain import initialized_repository

ROOT = Path(__file__).resolve().parents[1]


def config(lever):
    document = json.loads((ROOT / "configs/posttrain/secure-current.json").read_text())
    document["experiment"]["lever"] = lever
    return document


@pytest.mark.parametrize("lever", sorted(ONLINE_LEVERS))
def test_online_schema_accepts_each_objective_and_rejects_unknown_options(lever):
    document = config(lever)
    assert validate_config_dict(document) == []
    document["online"] = {"invented_optimizer": "anything"}
    assert any("unknown config.online" in error for error in validate_config_dict(document))


def test_invalid_online_lengths_batches_and_nan_are_rejected():
    document = config("grpo")
    for options in [
        {"batch_size": 3, "num_generations": 2},
        {"max_prompt_length": 16384},
        {"temperature": float("nan")},
        {"batch_size": True},
    ]:
        document["online"] = options
        with pytest.raises(ValueError):
            online_settings(document)


@pytest.mark.parametrize("lever", sorted(ONLINE_LEVERS))
def test_online_objectives_require_human_policy_authorization(lever):
    policy = copy.deepcopy(load_policy())
    policy.document["training"]["allowed_methods"] = [
        method
        for method in policy.document["training"]["allowed_methods"]
        if method not in {"grpo", "rloo", "opd", "sdpo", "sdft"}
    ]
    with pytest.raises(ValueError, match="not authorized"):
        validate_online_policy(config(lever), policy)


def test_verifiers_require_actual_correctness():
    assert verify_response("20 + 22 = 42", "numeric", {"value": 42})[0] == 0
    assert verify_response("42", "numeric", {"value": 42})[0] == 1
    truth = {"schema": {"type": "object", "required": ["answer"]}, "value": {"answer": 42}}
    assert verify_response('{"answer": 7}', "json_schema", truth)[0] == 0
    assert verify_response('{"answer": 42}', "json_schema", truth)[0] == 1
    assert "Ungültiges JSON" in verify_response("kein JSON", "json_schema", truth)[1]
    assert verify_response("B vor A", "ordered_terms", {"terms": ["A", "B"]})[0] == 0
    assert verify_response("A vor B", "ordered_terms", {"terms": ["A", "B"]})[0] == 1


def test_secure_normalizer_preserves_prompt_and_verifier_labels():
    source = {
        "dataset_id": "openeurollm/test",
        "revision": "a" * 40,
        "config": "de",
        "split": "train",
        "license": "MIT",
    }
    row = normalize_row(
        {
            "prompt": "Antworte nur mit einer Zahl.",
            "task_type": "numeric",
            "ground_truth": {"value": 42},
        },
        source,
        "0",
    )
    assert row["type"] == "rlvr"
    assert "42" in row_texts(row)[1]
    assert row["source"] == {**{k: v for k, v in source.items() if k != "license"}, "row_id": "0"}
    with pytest.raises(DataError, match="invalid online"):
        normalize_row({"prompt": "Frage", "task_type": "unknown", "ground_truth": {}}, source, "1")


def test_self_teacher_context_is_privileged_and_does_not_mutate_student_prompt():
    row = {
        "prompt": [{"role": "user", "content": "Frage"}],
        "demonstration": "Antwort",
        "task_type": "exact",
        "ground_truth": {"value": "richtig"},
    }
    original = copy.deepcopy(row)
    assert "Antwort" in self_teacher_prompt(row, "sdft")[-1]["content"]
    feedback = self_teacher_prompt(row, "sdpo", "falsch")[-1]["content"]
    assert "falsch" in feedback and "richtig" in feedback
    assert row == original


def test_split_is_reproducible_disjoint_and_rejects_duplicate_ids():
    rows = [{"content_id": str(i)} for i in range(20)]
    train, validation = split_online_rows(rows, 0.1, 42)
    assert (train, validation) == split_online_rows(list(reversed(rows)), 0.1, 42)
    assert len(train) == 18 and len(validation) == 2
    assert not {r["content_id"] for r in train} & {r["content_id"] for r in validation}
    with pytest.raises(ValueError, match="duplicate"):
        split_online_rows([rows[0], rows[0]], 0.1, 42)


def test_reverse_kl_gradient_moves_student_toward_teacher_without_training_teacher():
    import torch

    student = torch.tensor([[2.0, -2.0], [1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0, 2.0], [0.0, 1.0]], requires_grad=True)
    loss, _, _ = reverse_kl(student, teacher, chunk_size=1)
    previous = loss.mean().item()
    loss.mean().backward()
    assert teacher.grad is None
    with torch.no_grad():
        student -= student.grad * 0.5
    assert reverse_kl(student, teacher, chunk_size=2)[0].mean().item() < previous
    assert torch.allclose(reverse_kl(teacher, teacher, chunk_size=1)[0], torch.zeros(2))


def test_ema_updates_only_trainable_student_parameters():
    import torch

    student, teacher = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    student.bias.requires_grad_(False)
    teacher.requires_grad_(False)
    old_weight, old_bias = teacher.weight.clone(), teacher.bias.clone()
    update_ema_teacher(student, teacher, 0.5)
    assert torch.allclose(teacher.weight, old_weight * 0.5 + student.weight * 0.5)
    assert torch.equal(teacher.bias, old_bias)


@pytest.mark.parametrize("lever", sorted(ONLINE_LEVERS))
def test_every_online_lever_dispatches_one_secure_candidate_with_same_deadline(
    monkeypatch, tmp_path, lever
):
    document = config(lever)
    policy = copy.deepcopy(load_policy())
    policy.document["training"]["allowed_methods"].extend(["grpo", "rloo", "opd", "sdpo", "sdft"])
    calls = []
    monkeypatch.setattr(
        loop,
        "train_online_candidate",
        lambda **kwargs: calls.append(kwargs) or {"status": "succeeded", "run_id": "one-candidate"},
    )
    monkeypatch.setattr(loop, "resolve_model", lambda **kwargs: "resolved-teacher")
    monkeypatch.setattr(loop, "_teacher_license", lambda *args: "Apache-2.0")
    deadline = time.monotonic() + 90
    result = loop._execute_lever(
        ExperimentConfig(tmp_path / "config.json", document),
        policy,
        {"shards": []},
        deadline=deadline,
        outputs_root=tmp_path / "outputs",
        repository_root=tmp_path,
        allow_gpu=True,
        allow_checkpoints=True,
    )
    assert result["run_id"] == "one-candidate" and len(calls) == 1
    assert calls[0]["deadline"] == deadline
    assert calls[0]["teacher_ref"] == ("resolved-teacher" if lever in {"opd", "distill"} else None)


def test_opd_rejects_the_unchanged_student_seed_before_loading_a_trainer(monkeypatch, tmp_path):
    from types import SimpleNamespace

    policy = load_policy()
    monkeypatch.setattr(
        loop,
        "resolve_model",
        lambda **kwargs: SimpleNamespace(
            kind="hub_model",
            base_model={key: policy.seed_model[key] for key in ("repo_id", "revision")},
        ),
    )
    monkeypatch.setattr(
        loop, "_teacher_license", lambda *args: pytest.fail("must reject before Hub lookup")
    )
    monkeypatch.setattr(
        loop, "train_online_candidate", lambda **kwargs: pytest.fail("must not train")
    )
    with pytest.raises(loop.LoopError, match="distinct teacher"):
        loop._execute_lever(
            ExperimentConfig(tmp_path / "config.json", config("opd")),
            policy,
            {"shards": []},
            deadline=time.monotonic() + 90,
            outputs_root=tmp_path / "outputs",
            repository_root=tmp_path,
            allow_gpu=True,
            allow_checkpoints=True,
        )


@pytest.mark.parametrize("verb", ["grpo", "rlvr", "opd", "sdpo", "sdft", "distill"])
def test_manual_online_verbs_use_the_secure_producer(verb):
    args = cli.build_parser().parse_args(
        ["train", verb, "--real", "--allow-gpu", "--allow-checkpoints"]
    )
    assert args.handler is cli._train_command
    assert args.config.endswith("secure-current.json")


def _fixture_inputs(tmp_path, tiny_model_dir, lever):
    repository = initialized_repository(tmp_path / "repo")
    # More space for the feedback-conditioned prompt than the default tiny fixture.
    model_config_path = tiny_model_dir / "config.json"
    model_config = json.loads(model_config_path.read_text())
    model_config["max_position_embeddings"] = 256
    model_config_path.write_text(json.dumps(model_config))
    document = copy.deepcopy(load_policy().document)
    seed = document["seed_model"]
    seed.update(repo_id=str(tiny_model_dir), revision="a" * 40, context_length=256)
    for key, filename in {
        "model_config_sha256": "config.json",
        "tokenizer_sha256": "tokenizer.json",
        "tokenizer_config_sha256": "tokenizer_config.json",
        "chat_template_sha256": "chat_template.jinja",
    }.items():
        seed[key] = sha256_file(tiny_model_dir / filename)
    document["training"]["allowed_methods"].extend(["grpo", "rloo", "opd", "sdpo", "sdft"])
    policy_path = repository / "policy.json"
    policy_path.write_text(json.dumps(document))
    policy = load_policy(policy_path)
    cfg = config(lever)
    cfg["training"].update(
        method="lora",
        context_length=256,
        max_steps=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    cfg["distillation"].update(max_new_tokens=8, min_new_tokens=0, max_prompts=10)
    cfg["online"] = {"max_prompt_length": 128, "max_completion_length": 8}
    if lever in {"grpo", "rlvr", "sdpo"}:
        cfg["online"]["reward_profile"] = "reference_exact"
    if lever in {"grpo", "rlvr"}:
        cfg["online"].update(batch_size=2, num_generations=2, beta=0.0)
    path = repository / "training.jsonl"
    rows = [
        {
            "type": "sft",
            "content_id": f"case-{i}",
            "messages": [
                {"role": "user", "content": f"Frage {i}"},
                {"role": "assistant", "content": "Antwort richtig"},
            ],
        }
        for i in range(6)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    shard = ArtifactRef.from_path(
        path, role="sft_shard", media_type="application/jsonl", relative_to=repository
    ).to_dict()
    manifest = {
        "status": "trainable",
        "policy_sha256": sha256_file(policy.path),
        "license_status": "usable",
        "leakage_statistics": {"status": "clean", "hit_count": 0},
        "shards": [shard],
        "reports": [],
    }
    teacher_directory = tiny_model_dir
    if lever == "opd":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        teacher_directory = repository / "distinct-teacher"
        teacher_model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
        with torch.no_grad():
            teacher_model.lm_head.weight.mul_(3.0)
        teacher_model.save_pretrained(teacher_directory)
        AutoTokenizer.from_pretrained(tiny_model_dir).save_pretrained(teacher_directory)
    teacher_ref = ResolvedModelRef(
        kind="local_full_checkpoint",
        requested=str(teacher_directory),
        base_model={"repo_id": str(tiny_model_dir), "revision": "a" * 40},
        artifact=ArtifactRef.from_path(
            teacher_directory,
            role="full_checkpoint",
            media_type="application/vnd.boldt.transformers-checkpoint",
        ).to_dict(),
        tokenizer_sha256=seed["tokenizer_sha256"],
        chat_template_sha256=seed["chat_template_sha256"],
        model_config_sha256=seed["model_config_sha256"],
        architecture=seed["architecture"],
        source_run_id=None,
    )
    return repository, policy, cfg, manifest, teacher_ref


@pytest.mark.parametrize("lever", ["grpo", "rlvr", "opd", "sdpo", "sdft"])
def test_real_cuda_online_candidate_refreshes_rollouts_saves_reloads_and_resolves(
    tmp_path, tiny_model_dir, lever
):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("explicit CUDA integration fixture; no CPU fallback")
    repository, policy, cfg, manifest, teacher_ref = _fixture_inputs(
        tmp_path, tiny_model_dir, lever
    )
    outputs = repository / "outputs/posttrain"
    result = train_online_candidate(
        config=cfg,
        policy=policy,
        manifest=manifest,
        outputs_root=outputs,
        repository_root=repository,
        deadline=time.monotonic() + 120,
        allow_gpu=True,
        allow_checkpoints=True,
        teacher_ref=teacher_ref if lever == "opd" else None,
        teacher_license="Apache-2.0" if lever == "opd" else None,
    )
    assert result["status"] == "succeeded"
    assert result["metrics"]["steps_completed"] == 2
    resolved = resolve_candidate(result["run_id"], policy, outputs_root=outputs)
    assert resolved.kind == "peft_adapter"
    records = [
        json.loads(line)
        for line in (outputs / "runs" / result["run_id"] / "rollouts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {0, 1} <= {r["rollout_policy_step"] for r in records}
    assert all(record["completion_ids"] for record in records)
    if lever in {"opd", "sdpo", "sdft"}:
        from safetensors.torch import load_file

        assert all(
            len(r["completion_ids"]) == len(r["teacher_log_probs"]) == len(r["token_reverse_kl"])
            for r in records
        )
        weights = load_file(
            outputs / "checkpoints" / result["run_id"] / "adapter_model.safetensors"
        )
        assert any(
            tensor.abs().max().item() > 0 for name, tensor in weights.items() if "lora_B" in name
        ), "dense online signals must actually update the initially zero LoRA B matrices"
    EventLog(outputs).validate()
    assert not (outputs / "data").exists(), "no offline teacher dataset may be produced"


def test_online_rows_reject_tampered_training_shards(tmp_path):
    cfg = config("sdft")
    path = tmp_path / "source.jsonl"
    path.write_text("original")
    manifest = {
        "shards": [
            ArtifactRef.from_path(path, role="sft_shard", media_type="application/jsonl").to_dict()
        ]
    }
    path.write_text("tampered")
    with pytest.raises(ValueError, match="hash|size"):
        online_rows(manifest, cfg, root=tmp_path)


@pytest.mark.parametrize("lever", ["grpo", "rlvr"])
def test_native_rl_updates_adapter_on_mixed_verifier_rewards(
    tmp_path, tiny_model_dir, monkeypatch, lever
):
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaForCausalLM

    if not torch.cuda.is_available():
        pytest.skip("explicit CUDA optimizer fixture; no CPU fallback")
    repository, policy, cfg, manifest, _ = _fixture_inputs(tmp_path, tiny_model_dir, lever)

    # Control only the sampled answers: each prompt group contains one verified
    # answer and one wrong answer, both terminated. The native TRL objective,
    # log probabilities, gradients, optimizer, publication and reload stay real.
    def mixed_answers(self, input_ids, **kwargs):
        suffix = torch.tensor(
            [[7, 12 if index % 2 == 0 else 13, 1] for index in range(input_ids.shape[0])],
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix], dim=-1)

    monkeypatch.setattr(LlamaForCausalLM, "generate", mixed_answers)
    outputs = repository / "outputs/posttrain"
    result = train_online_candidate(
        config=cfg,
        policy=policy,
        manifest=manifest,
        outputs_root=outputs,
        repository_root=repository,
        deadline=time.monotonic() + 120,
        allow_gpu=True,
        allow_checkpoints=True,
    )
    weights = load_file(outputs / "checkpoints" / result["run_id"] / "adapter_model.safetensors")
    assert any(t.abs().max().item() > 0 for name, t in weights.items() if "lora_B" in name)
    records = [
        json.loads(line)
        for line in (outputs / "runs" / result["run_id"] / "rollouts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {record["reward"] for record in records} == {0.0, 1.0}
    assert {record["phase"] for record in records} == {"train", "validation"}
