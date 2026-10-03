"""Compatibility entrypoints delegate to the canonical CLI without changing status."""

import importlib.util
from pathlib import Path

import pytest

from boldt_posttrain import cli


@pytest.mark.parametrize(
    ("script", "prefix"),
    [
        ("pt_baseline", ["baseline", "run"]),
        ("pt_eval", ["eval", "run"]),
        ("pt_score", ["score"]),
        ("pt_promote", ["promote"]),
        ("pt_status", ["status"]),
        ("pt_report", ["report"]),
        ("pt_frontier_status", ["status"]),
        ("pt_merge_search", ["merge", "search"]),
        ("pt_discover_openeurollm_de", ["data", "discover"]),
        ("pt_prepare_openeurollm_de", ["data", "prepare"]),
        ("pt_loop", ["loop", "run"]),
        ("pt_distill_trial", ["distill"]),
    ],
)
def test_scripts_forward_exact_arguments_and_nonzero_exit_code(monkeypatch, script, prefix):
    seen = []
    monkeypatch.setattr(cli, "main", lambda argv: seen.extend(argv) or 5)
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(script, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--candidate", "exact-run-id"]) == 5
    assert seen == [*prefix, "--candidate", "exact-run-id"]
