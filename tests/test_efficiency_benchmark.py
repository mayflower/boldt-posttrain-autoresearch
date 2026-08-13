from boldt_posttrain.training import compare_sft_rlvr


def test_equal_budget_comparison_reports_both_winners():
    def sft(**_kwargs):
        return {"status": "ok", "quality": 0.6, "gpu_minutes": 10, "peak_vram": 10, "tokens": 1000}

    def rlvr(**_kwargs):
        return {
            "status": "ok",
            "quality": 0.7,
            "gpu_minutes": 10,
            "peak_vram": 11,
            "tokens": 500,
            "reward_development": [0.1, 0.4],
        }

    report = compare_sft_rlvr(
        sft_run=sft, rlvr_run=rlvr, model_start="m", prompt_group="g", gpu_minutes=10
    )
    assert report["quality_winner"] == "rlvr"
    assert report["efficiency_winner"] == "rlvr"
    assert report["reward_development"] == [0.1, 0.4]
