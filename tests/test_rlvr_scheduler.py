import pytest

from boldt_posttrain.scheduler import validate_search_plan


def test_rlvr_search_space_is_bounded():
    plan = {
        "trials": [
            {
                "trial_id": "r",
                "overrides": {"learning_rate": 1e-6, "lora_r": 8, "kl_coefficient": 0.05},
            }
        ]
    }
    assert validate_search_plan(plan, lever="rlvr")[0]["trial_id"] == "r"
    plan["trials"][0]["overrides"]["packing"] = True
    with pytest.raises(ValueError):
        validate_search_plan(plan, lever="rlvr")
