from boldt_posttrain.failure_mining import mine_failures


def test_fixed_taxonomy_keeps_technical_errors_separate():
    result = mine_failures(
        {"status": "failed", "run_id": "e1"},
        [
            {
                "case_id": "1",
                "category": "reasoning",
                "correct": False,
                "output": "x",
                "validator_errors": ["wrong"],
            },
            {"case_id": "2", "technical_error": "generation_error"},
        ],
        dimension_weights={"reasoning": 2},
    )
    assert result["categories"]["reasoning"]["priority"] == 2
    assert result["technical_errors"] == {"generation_error": 1}
    assert "prompt" not in result
