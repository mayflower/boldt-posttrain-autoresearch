from boldt_posttrain.evaluation import finalize_summary
from boldt_posttrain.frontier import update_specialist_frontiers, verified_merge_inputs


def test_specialist_can_advance_without_general_promotion():
    summary = {
        "run_id": "r",
        "model": "m",
        "mode": "real",
        "status": "ok",
        "technical_error_count": 0,
        "hard_gates": {"language": True, "safety": True, "format": True},
        "metrics": {
            "reasoning_core": 0.8,
            "leakage": {"status": "clean", "hits": 0},
            "license": {"usable": True},
        },
    }
    frontier = update_specialist_frontiers([finalize_summary(summary)])
    inputs = verified_merge_inputs(frontier)
    assert any(item["frontier"] == "reasoning" for item in inputs)
