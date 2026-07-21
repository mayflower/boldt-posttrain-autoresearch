import json
from pathlib import Path

from boldt_posttrain.artifacts import verify_artifact_ref
from boldt_posttrain.evaluation import _publish_evaluation
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import ResolvedModelRef


def test_baseline_pointer_and_artifact_refs_are_immutable(tmp_path: Path, monkeypatch):
    import boldt_posttrain.evaluation as evaluation

    policy = load_policy()
    resolved = ResolvedModelRef(
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
    baseline_root = tmp_path / "outputs/posttrain/baseline"
    result = _publish_evaluation(
        resolved=resolved,
        policy=policy,
        config_path=evaluation.ROOT / "configs/posttrain/current.json",
        output_root=baseline_root,
        baseline=True,
        replace_baseline=False,
        device="cpu",
    )
    pointer = json.loads((baseline_root / "current.json").read_text())
    assert pointer["run_id"] == result["run_id"]
    summary = json.loads((baseline_root / result["run_id"] / "summary.json").read_text())
    verify_artifact_ref(summary["raw_generations"], root=tmp_path)
    verify_artifact_ref(summary["model_artifact"], root=tmp_path)
    verify_artifact_ref(summary["lm_eval_artifact"], root=tmp_path)

    try:
        _publish_evaluation(
            resolved=resolved,
            policy=policy,
            config_path=evaluation.ROOT / "configs/posttrain/current.json",
            output_root=baseline_root,
            baseline=True,
            replace_baseline=False,
            device="cpu",
        )
    except evaluation.EvaluationError as exc:
        assert "replace-baseline" in str(exc)
    else:
        raise AssertionError("baseline replacement must be explicit")
