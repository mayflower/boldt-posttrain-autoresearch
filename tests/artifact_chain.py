import copy
import subprocess
from pathlib import Path

from boldt_posttrain import provenance
from boldt_posttrain.artifacts import (
    ArtifactRef,
    EventLog,
    atomic_write_json,
    canonical_json_bytes,
    new_run_id,
    sha256_bytes,
    sha256_file,
)
from boldt_posttrain.evaluation import _publish_evaluation
from boldt_posttrain.policy import Policy, load_policy
from boldt_posttrain.resolver import ResolvedModelRef


def initialized_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "fixture"], cwd=path, check=True)
    return path


def fixture_policy(tmp_path: Path) -> Policy:
    document = copy.deepcopy(load_policy().document)
    document["evaluation"]["bootstrap"]["samples"] = 200
    path = tmp_path / "policy.json"
    atomic_write_json(path, document)
    return load_policy(path)


def _candidate_run(repository: Path, policy: Policy) -> tuple[str, ResolvedModelRef]:
    outputs = repository / "outputs/posttrain"
    run_id = new_run_id("train-sft")
    checkpoint = outputs / "checkpoints" / run_id
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
    checkpoint_ref = ArtifactRef.from_path(
        checkpoint,
        role="adapter_checkpoint",
        media_type="application/vnd.boldt.peft-adapter",
        relative_to=repository,
    )
    model = {
        "base_model": {
            "repo_id": policy.seed_model["repo_id"],
            "revision": policy.seed_model["revision"],
        },
        "tokenizer_sha256": policy.seed_model["tokenizer_sha256"],
        "chat_template_sha256": policy.seed_model["chat_template_sha256"],
        "model_config_sha256": policy.seed_model["model_config_sha256"],
        "architecture": policy.seed_model["architecture"],
    }
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
        "command": ["python", "-m", "boldt_posttrain.cli", "train", "sft", "--real"],
        "git": provenance.collect_git("HEAD", root=repository),
        "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
        "experiment": {
            "path": "fixture",
            "sha256": sha256_bytes(b"fixture"),
            "resolved_sha256": sha256_bytes(b"fixture"),
        },
        "inputs": [],
        "outputs": [checkpoint_ref.to_dict()],
        "model": model,
        "data": {
            "license_status": "usable",
            "leakage_statistics": {"status": "clean", "hit_count": 0},
        },
        "parameters": {},
        "hardware": {},
        "environment": {
            "event_head": {key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")}
        },
        "parents": [],
        "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(model)),
        "error": None,
    }
    card_path = outputs / "runs" / run_id / "run_card.json"
    atomic_write_json(card_path, card)
    events.append(
        "run_finished",
        run_id,
        {"status": "succeeded", "run_card_sha256": sha256_file(card_path)},
    )
    resolved = ResolvedModelRef(
        kind="peft_adapter",
        requested=run_id,
        base_model=model["base_model"],
        artifact=checkpoint_ref.to_dict(),
        tokenizer_sha256=model["tokenizer_sha256"],
        chat_template_sha256=model["chat_template_sha256"],
        model_config_sha256=model["model_config_sha256"],
        architecture=model["architecture"],
        source_run_id=run_id,
    )
    return run_id, resolved


def _baseline_model(policy: Policy) -> ResolvedModelRef:
    return ResolvedModelRef(
        kind="hub_model",
        requested=policy.seed_model["repo_id"],
        base_model={
            "repo_id": policy.seed_model["repo_id"],
            "revision": policy.seed_model["revision"],
        },
        artifact=None,
        tokenizer_sha256=policy.seed_model["tokenizer_sha256"],
        chat_template_sha256=policy.seed_model["chat_template_sha256"],
        model_config_sha256=policy.seed_model["model_config_sha256"],
        architecture=policy.seed_model["architecture"],
        source_run_id=None,
    )


def complete_chain(tmp_path: Path, monkeypatch, *, improvement: float = 0.1) -> dict:
    import boldt_posttrain.evaluation as evaluation

    repository = initialized_repository(tmp_path / "repo")
    outputs = repository / "outputs/posttrain"
    policy = fixture_policy(tmp_path)
    candidate_run_id, candidate_model = _candidate_run(repository, policy)
    baseline_model = _baseline_model(policy)

    def records(resolved, cases, *, device):
        candidate = resolved.kind == "peft_adapter"
        result = []
        for case in cases:
            score = 0.5
            if candidate and case["category"] == "german_instruction":
                score += improvement
            result.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "prompt": case["prompt"],
                    "output": "Buch",
                    "score": score,
                    "validator_detail": {
                        "empty": False,
                        "refusal": False,
                        "english_bleed": False,
                    },
                    "error": None,
                }
            )
        return result

    monkeypatch.setattr(evaluation, "generate_cases", records)
    monkeypatch.setattr(
        evaluation,
        "run_lm_eval",
        lambda *args, **kwargs: {
            task: 0.5 for task in policy.document["evaluation"]["lm_eval_tasks"]
        },
    )
    baseline_result = _publish_evaluation(
        resolved=baseline_model,
        policy=policy,
        config_path=evaluation.ROOT / "configs/posttrain/current.json",
        output_root=outputs / "baseline",
        baseline=True,
        replace_baseline=False,
        device="cpu",
        repository_root=repository,
    )
    candidate_result = _publish_evaluation(
        resolved=candidate_model,
        policy=policy,
        config_path=evaluation.ROOT / "configs/posttrain/current.json",
        output_root=outputs / "evals",
        baseline=False,
        replace_baseline=False,
        device="cpu",
        repository_root=repository,
    )
    return {
        "repository": repository,
        "outputs": outputs,
        "policy": policy,
        "candidate_run_id": candidate_run_id,
        "candidate_eval_run_id": candidate_result["run_id"],
        "baseline_eval_run_id": baseline_result["run_id"],
    }
