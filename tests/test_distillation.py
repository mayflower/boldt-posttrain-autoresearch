import json
import subprocess
from pathlib import Path

import pytest

from boldt_posttrain.artifacts import verify_artifact_ref
from boldt_posttrain.distillation import DistillationError, distill_and_train
from boldt_posttrain.policy import load_policy
from boldt_posttrain.resolver import resolve_model
from tests.test_training_preflight import experiment
from tests.tiny_model import build_tiny_model


class GermanFixtureLanguage:
    def check(self, text: str) -> tuple[bool, float]:
        return (bool(text.strip()), 1.0)


def test_distillation_requires_checkpoint_permission(tmp_path: Path):
    model_path = build_tiny_model(tmp_path / "tiny")
    policy = load_policy()
    teacher = resolve_model(
        policy=policy,
        model=str(model_path),
        external_roots=(tmp_path,),
    )
    with pytest.raises(DistillationError, match="checkpoint permission"):
        distill_and_train(
            teacher=teacher,
            teacher_license="Apache-2.0",
            student_model_source=str(model_path),
            student_model_revision=None,
            prompts=["Erkläre ein Buch in einem deutschen Satz."],
            output_data_root=tmp_path / "outputs/posttrain/data",
            output_checkpoint_root=tmp_path / "outputs/posttrain/checkpoints",
            policy=policy,
            training=experiment(),
            generation={"min_new_tokens": 4, "max_new_tokens": 8},
            target_modules=["c_attn"],
            device="cpu",
            qlora=False,
            allow_checkpoints=False,
            budget_minutes=2,
            language_identifier=GermanFixtureLanguage(),
        )


def test_local_teacher_generation_is_gated_and_trains_student(tmp_path: Path, monkeypatch):
    import boldt_posttrain.training as training_module

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"], cwd=repository, check=True
    )
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "fixture"], cwd=repository, check=True)
    state_root = repository / "outputs/posttrain"
    model_path = build_tiny_model(tmp_path / "tiny")
    policy = load_policy()
    teacher = resolve_model(
        policy=policy,
        model=str(model_path),
        external_roots=(tmp_path,),
    )
    monkeypatch.setattr(training_module, "OUTPUTS", state_root)
    result = distill_and_train(
        teacher=teacher,
        teacher_license="Apache-2.0",
        student_model_source=str(model_path),
        student_model_revision=None,
        prompts=[
            "Erkläre ein Buch in einem deutschen Satz.",
            "Beschreibe Regen kurz und sachlich auf Deutsch.",
        ],
        output_data_root=state_root / "data",
        output_checkpoint_root=state_root / "checkpoints",
        policy=policy,
        training=experiment(),
        generation={"min_new_tokens": 4, "max_new_tokens": 8},
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
        repository_root=repository,
        language_identifier=GermanFixtureLanguage(),
    )
    assert result["status"] == "succeeded"
    manifest = json.loads(Path(result["teacher_data_manifest"]).read_text())
    assert manifest["status"] == "trainable"
    assert manifest["leakage_statistics"] == {"status": "clean", "hit_count": 0}
    assert manifest["teacher"]["artifact"]["sha256"] == teacher.artifact["sha256"]
    for ref in [*manifest["shards"], *manifest["reports"]]:
        verify_artifact_ref(ref, root=repository)
    raw_ref = next(ref for ref in manifest["reports"] if ref["role"] == "teacher_generations")
    raw_path = verify_artifact_ref(raw_ref, root=repository)
    generations = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert all(item["output"].startswith("Buch") for item in generations)
    student_card = json.loads(
        (state_root / "runs" / result["student_run_id"] / "run_card.json").read_text()
    )
    assert student_card["parents"] == [result["teacher_data_run_id"]]
    assert student_card["parameters"]["lineage"]["teacher_data_sha256"] == next(
        ref["sha256"] for ref in student_card["inputs"] if ref["role"] == "data_manifest"
    )
