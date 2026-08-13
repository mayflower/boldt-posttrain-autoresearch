import copy

from boldt_posttrain.scoring import score_run


def document(profile="promotion"):
    return {
        "status": "ok",
        "mode": "real",
        "profile": profile,
        "technical_error_count": 0,
        "metrics": {
            "german_instruction": 0.8,
            "format_following": 0.9,
            "reasoning_core": 0.7,
            "longcontext": 0.7,
            "safety": 0.95,
            "german_language_retention": 0.99,
            "english_bleed_rate": 0.01,
            "empty_output_rate": 0.0,
            "refusal_rate": 0.1,
            "over_refusal_rate": 0.02,
            "lm_eval": {"arc_de": 0.5},
            "leakage": {"status": "clean", "hits": 0},
            "license": {"status": "apache-2.0", "usable": True},
        },
    }


def test_promotion_requires_all_confidence_bounds():
    baseline = document()
    run = copy.deepcopy(baseline)
    run["metrics"]["german_instruction"] = 0.82
    run["confidence_intervals"] = {
        "german_instruction": {"lower": 0.01, "upper": 0.03},
        "safety": {"lower": 0.0, "upper": 0.01},
        "over_refusal_rate": {"lower": 0.0, "upper": 0.01},
        "english_bleed_rate": {"lower": 0.0, "upper": 0.01},
        "german_language_retention": {"lower": 0.0, "upper": 0.01},
    }
    assert score_run(run, baseline)["status"] == "pass"
    del run["confidence_intervals"]["safety"]
    assert "safety_ci" in [gate["name"] for gate in score_run(run, baseline)["failed_gates"]]
