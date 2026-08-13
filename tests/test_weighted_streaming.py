from boldt_posttrain.data_pipeline import deterministic_weighted_interleave


def test_streaming_never_repeats_unless_explicit():
    groups = {"a": [{"x": 1}], "b": [{"x": 2}, {"x": 3}]}
    rows = list(
        deterministic_weighted_interleave(
            groups, {"a": 0.5, "b": 0.5}, seed=7, token_count=lambda _row: 1
        )
    )
    assert len(rows) == 3
    assert all(row["_mix_metrics"]["repeat_count"] == 0 for _, row in rows)
