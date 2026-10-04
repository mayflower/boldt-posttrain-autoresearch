"""Sequence-level distillation: policy teachers, generation gates, and the seqkd lever."""

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from boldt_posttrain import cli, loop, seqkd
from boldt_posttrain.artifacts import ArtifactRef, EventLog, sha256_file
from boldt_posttrain.policy import PolicyError, load_policy, validate_policy
from boldt_posttrain import data_pipeline
from boldt_posttrain.config import ExperimentConfig, validate_config_dict
from boldt_posttrain.data_pipeline import normalize_row
from tests.artifact_chain import initialized_repository

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/posttrain/experiments/seqkd-qwen3.8-27b-de.json"
TEACHER = "Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"


def experiment() -> dict:
    return json.loads(EXPERIMENT.read_text())


# -- policy ------------------------------------------------------------------------


def test_repository_policy_approves_exactly_the_pinned_qwen_teacher():
    policy = load_policy()
    assert "seqkd" in policy.document["training"]["allowed_methods"]
    assert seqkd.policy_teacher(policy, TEACHER) == {
        "repo_id": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "license": "Apache-2.0",
        "purpose": "seqkd",
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"revision": "main"}, "exact 40-character commit"),
        ({"license": "llama3"}, "allowed_licenses"),
        ({"purpose": "opd"}, "purpose"),
        ({"repo_id": "mayflowergmbh/boldt-dc-1b-german-it-16k-dpo"}, "differ from the student"),
        ({"extra": "x"}, "unknown policy.teachers"),
    ],
)
def test_policy_rejects_movable_unlicensed_or_seed_teachers(change, message):
    document = copy.deepcopy(load_policy().document)
    document["teachers"][0].update(change)
    with pytest.raises(PolicyError, match=message):
        validate_policy(document)


def test_policy_rejects_duplicate_teacher_entries_and_missing_section():
    document = copy.deepcopy(load_policy().document)
    document["teachers"].append(dict(document["teachers"][0]))
    with pytest.raises(PolicyError, match="duplicates"):
        validate_policy(document)
    del document["teachers"]
    with pytest.raises(PolicyError, match="missing policy.teachers"):
        validate_policy(document)


def test_experiment_cannot_name_a_teacher_outside_policy():
    other = "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    with pytest.raises(seqkd.SeqKDError, match="not an approved entry"):
        seqkd.policy_teacher(load_policy(), other)


# -- experiment schema ------------------------------------------------------------


def test_seqkd_experiment_file_is_strictly_valid():
    assert validate_config_dict(experiment()) == []
    settings = seqkd.validate_seqkd_policy(experiment(), load_policy())
    assert settings["teacher"] == TEACHER and settings["generation_run"] == ""


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["seqkd"].update(extra=1), "unknown=\\['extra'\\]"),
        (lambda d: d["seqkd"].pop("seed"), "missing=\\['seed'\\]"),
        (lambda d: d["seqkd"].update(teacher="Qwen/Qwen3.8-27B-FP8"), "40-character"),
        (lambda d: d["seqkd"].update(top_p=0.0), "top_p"),
        (lambda d: d["seqkd"].update(max_prompts=True), "invalid type"),
        (lambda d: d["seqkd"].update(generation_run="latest"), "exact run ID"),
        (lambda d: d.pop("seqkd"), "config.seqkd must be an object"),
    ],
)
def test_seqkd_block_is_exact(mutate, message):
    document = experiment()
    mutate(document)
    errors = validate_config_dict(document)
    assert any(__import__("re").search(message, error) for error in errors), errors


def test_existing_experiments_without_seqkd_stay_valid():
    document = json.loads((ROOT / "configs/posttrain/secure-current.json").read_text())
    assert "seqkd" not in document and validate_config_dict(document) == []


# -- prompt selection and answer gates --------------------------------------------


def _row(index: int, *, system: bool = False) -> dict:
    messages = [{"role": "user", "content": f"Erkläre bitte Thema Nummer {index} ausführlich."}]
    if system:
        messages.insert(0, {"role": "system", "content": "Du bist ein hilfreicher Assistent."})
    messages.append({"role": "assistant", "content": "Ursprüngliche Antwort aus dem Datensatz."})
    source = {
        "dataset_id": "openeurollm/EU-Instruct-Synthetic",
        "revision": "c" * 40,
        "config": "de",
        "split": "train",
        "license": "Apache-2.0",
    }
    return normalize_row({"messages": messages}, source, str(index))


def test_prompt_prefix_keeps_context_up_to_first_assistant_turn():
    row = _row(1, system=True)
    assert [item["role"] for item in seqkd.prompt_prefix(row["messages"])] == ["system", "user"]
    assert seqkd.prompt_prefix([{"role": "assistant", "content": "x"}]) is None


def test_prompt_selection_is_deterministic_and_order_independent():
    rows = [_row(index) for index in range(20)]
    first = seqkd.select_prompts(rows, maximum=5, seed=7)
    second = seqkd.select_prompts(list(reversed(rows)), maximum=5, seed=7)
    assert [row["content_id"] for row, _ in first] == [row["content_id"] for row, _ in second]
    assert len(seqkd.select_prompts(rows, maximum=50, seed=7)) == 20


@pytest.mark.parametrize(
    ("generation", "reason"),
    [
        ({"text": "<think>Hmm</think> Antwort", "finish_reason": "stop"}, "thinking_leak"),
        ({"text": "Eine abgeschnittene Antwort", "finish_reason": "length"}, "truncated"),
        ({"text": "   ", "finish_reason": "stop"}, "empty"),
        ({"text": "", "finish_reason": "prompt_too_long"}, "prompt_too_long"),
        ({"text": "Eine vollständige Antwort.", "finish_reason": "stop"}, None),
    ],
)
def test_answer_gates(generation, reason):
    assert seqkd.rejection_reason(generation) == reason


# -- end-to-end generation with an injected teacher -------------------------------


class GermanOnly:
    """Deterministic stand-in for the fastText gate: rejects answers marked English."""

    def check(self, text):
        return ("ENGLISH" not in text, 0.99)


def _prepared_repository(tmp_path, monkeypatch, rows):
    repository = initialized_repository(tmp_path / "repo")
    outputs = repository / "outputs/posttrain"
    shard_path = outputs / "data/prepare-fixture/train_sft-00000-of-00001.jsonl"
    shard_path.parent.mkdir(parents=True)
    shard_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    shard = ArtifactRef.from_path(
        shard_path, role="sft_shard", media_type="application/jsonl", relative_to=repository
    ).to_dict()
    manifest = {"run_id": "data-prepare-fixture", "status": "trainable", "shards": [shard]}
    monkeypatch.setattr(data_pipeline, "verify_data_manifest", lambda *args, **kwargs: manifest)
    return repository, outputs


def _teacher_ref():
    return SimpleNamespace(
        kind="hub_model",
        tokenizer_sha256="1" * 64,
        chat_template_sha256="2" * 64,
        model_config_sha256="3" * 64,
        architecture="Qwen3_5ForConditionalGeneration",
    )


def _generate(repository, outputs, generator, config_path=EXPERIMENT, license_id="Apache-2.0"):
    return seqkd.generate(
        load_policy(),
        config_path,
        outputs_root=outputs,
        repository_root=repository,
        generator=generator,
        resolve_teacher=lambda reference, policy: _teacher_ref(),
        teacher_license=lambda *args: license_id,
        language=GermanOnly(),
    )


def _answers(prompts, *, settings, teacher, max_model_len):
    assert teacher["repo_id"] == "Qwen/Qwen3.8-27B-FP8" and max_model_len == 16384
    assert settings["temperature"] == 0.7
    answers = []
    for index, prompt in enumerate(prompts):
        topic = prompt[-1]["content"]
        if index == 0:
            answers.append({"text": "<think>x</think>", "finish_reason": "stop"})
        elif index == 1:
            answers.append({"text": "ENGLISH answer only", "finish_reason": "stop"})
        else:
            answers.append(
                {
                    "text": f"Ausführliche deutsche Antwort des Lehrermodells zu: {topic}",
                    "finish_reason": "stop",
                    "prompt_tokens": 10,
                    "completion_tokens": 12,
                }
            )
    return answers


def _settings(**changes):
    settings = seqkd.seqkd_settings(experiment())
    settings.update(changes)
    return settings


def test_generation_publishes_gated_teacher_answers_and_feeds_the_lever(tmp_path, monkeypatch):
    repository, outputs = _prepared_repository(tmp_path, monkeypatch, [_row(i) for i in range(6)])
    result = _generate(repository, outputs, _answers)
    assert result["status"] == "succeeded" and result["rows_trainable"] == 4
    assert result["rejections"] == {"language_not_german": 1, "thinking_leak": 1}

    manifest, card = seqkd.verify_generation(
        _settings(generation_run=result["run_id"]),
        load_policy(),
        outputs_root=outputs,
        repository_root=repository,
    )
    assert card["run_type"] == "seqkd_generate" and card["parents"] == ["data-prepare-fixture"]
    assert manifest["teacher"]["revision"] == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    assert manifest["leakage_statistics"] == {"status": "clean", "hit_count": 0}
    rows = [
        json.loads(line)
        for line in (repository / manifest["shards"][0]["path"]).read_text().splitlines()
    ]
    assert all(row["messages"][-1]["content"].startswith("Ausführliche") for row in rows)
    assert all(row["teacher"]["repo_id"] == "Qwen/Qwen3.8-27B-FP8" for row in rows)

    document = experiment()
    document["seqkd"]["generation_run"] = result["run_id"]
    calls = []
    monkeypatch.setattr(
        loop,
        "train_adapter",
        lambda **kwargs: calls.append(kwargs) or {"status": "succeeded", "run_id": "student"},
    )
    lever = loop._execute_lever(
        ExperimentConfig(tmp_path / "config.json", document),
        load_policy(),
        {"shards": []},
        deadline=time.monotonic() + 90,
        outputs_root=outputs,
        repository_root=repository,
        allow_gpu=True,
        allow_checkpoints=True,
    )
    assert lever["run_id"] == "student" and len(calls) == 1
    call = calls[0]
    assert call["run_type"] == "train_seqkd" and call["kind"] == "sft"
    assert call["parent_run_ids"] == [result["run_id"]]
    assert call["lineage"]["seqkd_generation_run"] == result["run_id"]
    assert [ref["role"] for ref in call["input_artifacts"]] == ["sft_shard"]
    assert len(call["dataset"]) == 4


def test_benchmark_leakage_in_teacher_answers_publishes_nothing(tmp_path, monkeypatch):
    from boldt_posttrain.evaluation import load_suite

    leaked = load_suite()[0]["prompt"]
    repository, outputs = _prepared_repository(tmp_path, monkeypatch, [_row(i) for i in range(3)])

    def leaking(prompts, **kwargs):
        return [{"text": leaked, "finish_reason": "stop"} for _ in prompts]

    with pytest.raises(seqkd.SeqKDError, match="leakage"):
        _generate(repository, outputs, leaking)
    assert not list((outputs / "seqkd").glob("seqkd-generate-*"))
    assert not list((outputs / "runs").glob("seqkd-generate-*"))
    events = [json.loads(line) for line in EventLog(outputs).log_path.read_text().splitlines()]
    assert events[-1]["payload"] == {"status": "failed"}


def test_model_card_license_must_match_the_policy_entry(tmp_path, monkeypatch):
    repository, outputs = _prepared_repository(tmp_path, monkeypatch, [_row(1)])
    with pytest.raises(seqkd.SeqKDError, match="license differs"):
        _generate(repository, outputs, _answers, license_id="MIT")


def test_pinned_generation_fails_closed_on_drift_or_tamper(tmp_path, monkeypatch):
    repository, outputs = _prepared_repository(tmp_path, monkeypatch, [_row(i) for i in range(6)])
    run_id = _generate(repository, outputs, _answers)["run_id"]
    policy = load_policy()

    def verify(**changes):
        return seqkd.verify_generation(
            _settings(**{"generation_run": run_id, **changes}),
            policy,
            outputs_root=outputs,
            repository_root=repository,
        )

    with pytest.raises(seqkd.SeqKDError, match="parameters changed"):
        verify(temperature=0.2)
    with pytest.raises(seqkd.SeqKDError, match="is empty"):
        verify(generation_run="")
    with pytest.raises(seqkd.SeqKDError, match="no run card"):
        verify(generation_run="seqkd-generate-20260101T000000.000000Z-0123456789abcdef")
    shard = next((outputs / "seqkd" / run_id).glob("train_sft-*.jsonl"))
    shard.write_text(shard.read_text() + "{}\n")
    with pytest.raises(seqkd.SeqKDError, match="artifact verification failed"):
        verify()


def test_unpinned_seqkd_lever_refuses_to_train(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "train_adapter", lambda **kwargs: pytest.fail("must not train"))
    with pytest.raises(seqkd.SeqKDError, match="generation_run is empty"):
        loop._execute_lever(
            ExperimentConfig(tmp_path / "config.json", experiment()),
            load_policy(),
            {"shards": []},
            deadline=time.monotonic() + 90,
            outputs_root=tmp_path / "outputs",
            repository_root=tmp_path,
            allow_gpu=True,
            allow_checkpoints=True,
        )


# -- CLI --------------------------------------------------------------------------


def test_cli_exposes_generation_and_training_verbs():
    parser = cli.build_parser()
    generate = parser.parse_args(["seqkd", "generate", "--real", "--allow-gpu"])
    assert generate.handler is cli._seqkd_generate_command
    train = parser.parse_args(["train", "seqkd", "--real", "--allow-gpu", "--allow-checkpoints"])
    assert train.handler is cli._train_command
    with pytest.raises(cli.CliParseError):
        parser.parse_args(["seqkd", "generate"])


def test_opd_dry_run_rejects_the_student_seed_as_teacher(monkeypatch, capsys):
    monkeypatch.setattr(cli, "verify_data_manifest", lambda *args, **kwargs: {"shards": []})
    policy = copy.deepcopy(load_policy())
    policy.document["training"]["allowed_methods"].append("opd")
    monkeypatch.setattr(cli, "load_policy", lambda: policy)
    args = cli.build_parser().parse_args(["train", "opd", "--dry-run"])
    assert args.handler(args) == 2
    assert "distinct teacher" in json.loads(capsys.readouterr().out)["error"]


def test_real_generation_requires_explicit_gpu_permission(monkeypatch, capsys):
    monkeypatch.setattr(cli, "verify_data_manifest", lambda *args, **kwargs: {"shards": []})
    monkeypatch.setattr(seqkd, "generate", lambda *args, **kwargs: pytest.fail("must not run"))
    args = cli.build_parser().parse_args(
        ["seqkd", "generate", "--real", "--config", str(EXPERIMENT)]
    )
    assert args.handler(args) == 4
    assert "--allow-gpu" in json.loads(capsys.readouterr().out)["error"]


def test_generation_run_card_is_event_anchored(tmp_path, monkeypatch):
    repository, outputs = _prepared_repository(tmp_path, monkeypatch, [_row(i) for i in range(4)])
    run_id = _generate(repository, outputs, _answers)["run_id"]
    card = outputs / "runs" / run_id / "run_card.json"
    events = [json.loads(line) for line in EventLog(outputs).log_path.read_text().splitlines()]
    assert events[-1]["payload"] == {"status": "succeeded", "run_card_sha256": sha256_file(card)}
