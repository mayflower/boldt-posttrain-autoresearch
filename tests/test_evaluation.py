import json
from types import SimpleNamespace

from boldt_posttrain.evaluation import (
    EvaluationTechnicalError,
    deterministic_proxy_cases,
    evaluate_cases,
    refusal_metrics,
    run_lm_eval,
)


def test_tokenizer_failure_is_technical_not_model_error():
    cases = [{"case_id": "x", "category": "instruction", "expected": "ja"}]

    def broken(_case):
        raise EvaluationTechnicalError("tokenization_error", "bad tokenizer")

    result = evaluate_cases(cases, broken)
    assert result["status"] == "failed"
    assert result["technical_errors"] == {"tokenization_error": 1}
    assert result["model_error_count"] == 0


def test_proxy_is_deterministic_and_has_eight_per_category():
    cases = [
        {"case_id": f"{category}-{i}", "category": category}
        for category in ("a", "b")
        for i in range(12)
    ]
    first = deterministic_proxy_cases(cases)
    assert first == deterministic_proxy_cases(cases)
    assert len(first) == 16


def test_refusal_dimensions_are_separate():
    result = refusal_metrics(
        [
            {"output": "Ich kann dabei nicht helfen.", "should_refuse": True},
            {"output": "Ich kann dabei nicht helfen.", "should_refuse": False},
        ]
    )
    assert result == {"refusal_rate": 1.0, "desired_refusal_rate": 1.0, "over_refusal_rate": 1.0}


def test_lm_eval_receives_explicit_repaired_tokenizer(tmp_path, monkeypatch):
    captured = {}

    def run(command, **_kwargs):
        captured["command"] = command
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        (result_dir / "results.json").write_text(
            json.dumps({"results": {"task": {"acc,none": 1.0}}}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("boldt_posttrain.evaluation.subprocess.run", run)
    result = run_lm_eval(
        model="base/model",
        tokenizer_ref=str(tmp_path / "fixed-tokenizer"),
        tasks=["task"],
        output_path=tmp_path / "results",
        device="cpu",
    )
    model_args = captured["command"][captured["command"].index("--model_args") + 1]
    assert model_args == f"pretrained=base/model,tokenizer={tmp_path / 'fixed-tokenizer'}"
    assert result["metrics"] == {"task": 1.0}
