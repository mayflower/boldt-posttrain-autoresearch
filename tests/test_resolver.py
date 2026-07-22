import json
from pathlib import Path

import pytest

from boldt_posttrain.artifacts import (
    ArtifactRef,
    EventLog,
    atomic_write_json,
    new_run_id,
    sha256_file,
)
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import (
    CandidateRegistry,
    ResolutionError,
    load_tokenizer,
    resolve_candidate,
    resolve_model,
)
from tests.tiny_model import build_tiny_model


def successful_candidate(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    outputs = repo / "outputs/posttrain"
    checkpoint = outputs / "checkpoints" / "candidate"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}")
    ref = ArtifactRef.from_path(
        checkpoint,
        role="adapter_checkpoint",
        media_type="application/vnd.boldt.peft-adapter",
        relative_to=repo,
    )
    run_id = new_run_id("train-sft")
    policy = load_policy()
    events = EventLog(outputs)
    start = events.append("run_started", run_id, {"run_type": "train_sft"})
    card = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": "train_sft",
        "mode": "real",
        "status": "succeeded",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_seconds": 1.0,
        "command": ["python", "-m", "boldt_posttrain.cli", "train", "sft"],
        "git": {"base_ref": "a" * 40, "head": "a" * 40, "dirty": False, "diff_sha256": "b" * 64},
        "policy": {"path": "configs/posttrain/policy.json", "sha256": "c" * 64},
        "experiment": {
            "path": "configs/posttrain/current.json",
            "sha256": "d" * 64,
            "resolved_sha256": "e" * 64,
        },
        "inputs": [],
        "outputs": [ref.to_dict()],
        "model": {
            "base_model": {
                "repo_id": policy.seed_model["repo_id"],
                "revision": policy.seed_model["revision"],
            },
            "tokenizer_sha256": policy.seed_model["tokenizer_sha256"],
            "chat_template_sha256": policy.seed_model["chat_template_sha256"],
            "model_config_sha256": policy.seed_model["model_config_sha256"],
            "architecture": policy.seed_model["architecture"],
        },
        "data": {},
        "parameters": {},
        "hardware": {},
        "environment": {
            "event_head": {key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")}
        },
        "parents": [],
        "compatibility_fingerprint": "f" * 64,
        "error": None,
    }
    atomic_write_json(outputs / "runs" / run_id / "run_card.json", card)
    events.append(
        "run_finished",
        run_id,
        {
            "status": "succeeded",
            "run_card_sha256": sha256_file(outputs / "runs" / run_id / "run_card.json"),
        },
    )
    return outputs, run_id


def test_successful_adapter_resolves_exactly(tmp_path: Path):
    outputs, run_id = successful_candidate(tmp_path)
    resolved = resolve_candidate(run_id, load_policy(), outputs_root=outputs)
    assert resolved.kind == "peft_adapter"
    assert resolved.source_run_id == run_id
    assert resolved.base_model["revision"] == load_policy().seed_model["revision"]


def test_candidate_never_falls_back_to_seed_model(tmp_path: Path):
    outputs = tmp_path / "outputs/posttrain"
    with pytest.raises(ResolutionError, match="unknown or invalid candidate"):
        resolve_model(policy=load_policy(), candidate="DOES-NOT-EXIST", outputs_root=outputs)


@pytest.mark.parametrize("candidate", ["latest", "../escape", "/absolute", "a/b", "a\\b"])
def test_candidate_aliases_and_path_traversal_are_rejected(tmp_path: Path, candidate: str):
    with pytest.raises(ResolutionError):
        resolve_candidate(candidate, load_policy(), outputs_root=tmp_path / "outputs/posttrain")


def test_registry_contains_only_verified_successful_candidates(tmp_path: Path):
    outputs, run_id = successful_candidate(tmp_path)
    registry = CandidateRegistry(outputs).rebuild(load_policy())
    assert list(registry["candidates"]) == [run_id]
    assert (outputs / "registry/current.json").is_file()


def test_pointer_is_hash_verified(tmp_path: Path):
    outputs, run_id = successful_candidate(tmp_path)
    run_card = outputs / "runs" / run_id / "run_card.json"
    from boldt_posttrain.artifacts import sha256_file

    pointer = {
        "schema_version": 1,
        "pointer_id": "champion",
        "run_id": run_id,
        "run_card_sha256": sha256_file(run_card),
    }
    atomic_write_json(outputs / "registry/pointers/champion.json", pointer)
    assert (
        resolve_candidate("champion", load_policy(), outputs_root=outputs).source_run_id == run_id
    )
    run_card.write_text("{}")
    with pytest.raises(ResolutionError, match="hash mismatch"):
        resolve_candidate("champion", load_policy(), outputs_root=outputs)


def test_tokenizer_loader_overrides_invalid_extra_special_tokens_metadata(tmp_path: Path):
    model_path = build_tiny_model(tmp_path / "tiny")
    config_path = model_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["extra_special_tokens"] = ["<user>", "<assistant>"]
    config_path.write_text(json.dumps(config))

    tokenizer = load_tokenizer(model_path, local_files_only=True)

    assert tokenizer.convert_tokens_to_ids("<assistant>") == 4
    assert tokenizer.chat_template
