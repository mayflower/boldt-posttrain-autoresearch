import pytest

from boldt_posttrain.scheduler import build_mix_plan, run_mix_probes


def test_negative_utility_is_zero_and_mix_normalizes():
    plan = build_mix_plan(
        [
            {"source_group": "general", "proxy_score_delta": 2, "gpu_minutes": 1},
            {"source_group": "bad", "proxy_score_delta": -1, "gpu_minutes": 1},
            {"source_group": "safety", "proxy_score_delta": 1, "gpu_minutes": 1},
        ],
        minimum_weights={"safety": 0.2},
    )
    assert plan["weights"]["bad"] == 0
    assert sum(plan["weights"].values()) == pytest.approx(1)


def test_three_source_probes_are_serial_64_step_measurements():
    calls = []

    def probe(group, steps):
        calls.append((group, steps))
        return {
            "status": "ok",
            "proxy_score_delta": {"a": 1, "b": 2, "c": -1}[group],
            "gpu_minutes": 1,
        }

    plan = run_mix_probes(["c", "a", "b"], probe, minimum_weights={})
    assert calls == [("a", 64), ("b", 64), ("c", 64)]
    assert plan["weights"]["c"] == 0
    assert sum(plan["weights"].values()) == pytest.approx(1)
