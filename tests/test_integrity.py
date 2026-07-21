import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_integrity():
    spec = importlib.util.spec_from_file_location(
        "check_posttrain_integrity", ROOT / "scripts/check_posttrain_integrity.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "tests@example.invalid")
    git(repo, "config", "user.name", "Tests")
    (repo / "configs/posttrain").mkdir(parents=True)
    policy = json.loads((ROOT / "configs/posttrain/policy.json").read_text())
    policy_path = repo / "configs/posttrain/policy.json"
    policy_path.write_text(json.dumps(policy))
    (repo / "configs/posttrain/base.json").write_text("{}")
    (repo / "configs/posttrain/current.json").write_text("{}")
    (repo / "README.md").write_text("initial")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    return repo, policy_path


def test_integrity_rejects_every_non_editable_path_by_default():
    module = load_integrity()
    globs = module.load_globs()
    result = module.evaluate(["README.md"], globs=globs)
    assert result["status"] == "fail"
    assert result["other"] == ["README.md"]


def test_integrity_protects_policy_and_base_config():
    module = load_integrity()
    result = module.evaluate(
        ["configs/posttrain/policy.json", "configs/posttrain/base.json"], globs=module.load_globs()
    )
    assert result["status"] == "fail"
    assert result["protected"] == ["configs/posttrain/base.json", "configs/posttrain/policy.json"]


def test_changed_paths_combines_committed_uncommitted_and_untracked(repository: tuple[Path, Path]):
    repo, policy_path = repository
    module = load_integrity()
    base = git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("committed")
    git(repo, "add", "README.md")
    git(repo, "commit", "-qm", "committed change")
    (repo / "configs/posttrain/current.json").write_text('{"changed": true}')
    (repo / "untracked.txt").write_text("new")
    result = module.check(base_ref=base, root=repo, policy_path=policy_path)
    assert result["status"] == "fail"
    assert set(result["editable"]) == {"configs/posttrain/current.json"}
    assert set(result["other"]) == {"README.md", "untracked.txt"}


def test_integrity_fails_closed_on_git_error(
    repository: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    repo, policy_path = repository
    module = load_integrity()

    def broken(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(module.subprocess, "run", broken)
    result = module.check(root=repo, policy_path=policy_path)
    assert result["status"] == "fail"
    assert "git unavailable" in result["git_error"]


def test_invalid_base_ref_fails_closed(repository: tuple[Path, Path]):
    repo, policy_path = repository
    result = load_integrity().check(base_ref="DOES_NOT_EXIST", root=repo, policy_path=policy_path)
    assert result["status"] == "fail"
    assert result["git_error"]
