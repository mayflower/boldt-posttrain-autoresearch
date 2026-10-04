import json
from pathlib import Path

from boldt_posttrain.artifacts import verify_artifact_ref
from boldt_posttrain.evaluation import _publish_evaluation
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import ResolvedModelRef
from tests.artifact_chain import initialized_repository


def _seed(policy) -> ResolvedModelRef:
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


def _stub_generation(monkeypatch):
    import boldt_posttrain.evaluation as evaluation

    def records(_resolved, cases, *, device):
        return [
            {
                "case_id": item["case_id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "output": "Buch",
                "score": 0.5,
                "validator_detail": {"empty": False, "refusal": False, "english_bleed": False},
                "error": None,
            }
            for item in cases
        ]

    monkeypatch.setattr(evaluation, "generate_cases", records)
    monkeypatch.setattr(
        evaluation,
        "run_lm_eval",
        lambda *args, **kwargs: {
            task: 0.5 for task in load_policy().document["evaluation"]["lm_eval_tasks"]
        },
    )


def test_baseline_pointer_and_artifact_refs_are_immutable(tmp_path: Path, monkeypatch):
    import boldt_posttrain.evaluation as evaluation

    policy = load_policy()
    resolved = _seed(policy)
    _stub_generation(monkeypatch)
    repository = initialized_repository(tmp_path / "repo")
    baseline_root = repository / "outputs/posttrain/baseline"
    result = _publish_evaluation(
        resolved=resolved,
        policy=policy,
        config_path=evaluation.ROOT / "configs/posttrain/secure-current.json",
        output_root=baseline_root,
        baseline=True,
        replace_baseline=False,
        device="cpu",
        repository_root=repository,
    )
    pointer = json.loads((baseline_root / "current.json").read_text())
    assert pointer["run_id"] == result["run_id"]
    summary = json.loads((baseline_root / result["run_id"] / "summary.json").read_text())
    verify_artifact_ref(summary["raw_generations"], root=repository)
    verify_artifact_ref(summary["model_artifact"], root=repository)
    verify_artifact_ref(summary["lm_eval_artifact"], root=repository)

    try:
        _publish_evaluation(
            resolved=resolved,
            policy=policy,
            config_path=evaluation.ROOT / "configs/posttrain/secure-current.json",
            output_root=baseline_root,
            baseline=True,
            replace_baseline=False,
            device="cpu",
            repository_root=repository,
        )
    except evaluation.EvaluationError as exc:
        assert "replace-baseline" in str(exc)
    else:
        raise AssertionError("baseline replacement must be explicit")


def test_baseline_bound_to_an_older_policy_is_rebuilt_without_flag(tmp_path: Path, monkeypatch):
    import boldt_posttrain.evaluation as evaluation
    from boldt_posttrain.loop import readiness
    from tests.artifact_chain import fixture_policy

    _stub_generation(monkeypatch)
    repository = initialized_repository(tmp_path / "repo")
    baseline_root = repository / "outputs/posttrain/baseline"
    call = dict(
        config_path=evaluation.ROOT / "configs/posttrain/secure-current.json",
        output_root=baseline_root,
        baseline=True,
        replace_baseline=False,
        device="cpu",
        repository_root=repository,
    )
    old_policy = load_policy()
    first = _publish_evaluation(resolved=_seed(old_policy), policy=old_policy, **call)
    assert "supersedes_baseline" not in first

    new_policy = fixture_policy(tmp_path)  # different content, so a different policy hash
    stale = readiness(new_policy, outputs_root=baseline_root.parent, repository_root=repository)
    assert stale["baseline"]["status"] == "invalid"
    assert stale["next_command"] == "/pt-data real"  # no data manifest in this fixture

    second = _publish_evaluation(resolved=_seed(new_policy), policy=new_policy, **call)
    assert second["supersedes_baseline"]["run_id"] == first["run_id"]
    assert "policy hash" in second["supersedes_baseline"]["reason"]
    card = json.loads((baseline_root / second["run_id"] / "run_card.json").read_text())
    assert card["data"]["supersedes_baseline"]["run_id"] == first["run_id"]
    assert (baseline_root / first["run_id"] / "summary.json").is_file()
    fresh = readiness(new_policy, outputs_root=baseline_root.parent, repository_root=repository)
    assert fresh["baseline"] == {"status": "ok", "run_id": second["run_id"]}
