from boldt_posttrain.scheduler import continuation_metadata, rungs


def test_rungs_are_additional_and_continuation_is_honest():
    values = rungs(1000)
    assert [(r.additional_steps, r.cumulative_steps) for r in values] == [
        (64, 64),
        (192, 256),
        (744, 1000),
    ]
    assert continuation_metadata("parent") == {
        "parent_run_id": "parent",
        "continuation_mode": "adapter_fresh_optimizer",
    }
