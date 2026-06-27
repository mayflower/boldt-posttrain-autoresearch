"""Fail-closed behaviour of the protected scorer (pure stdlib unittest)."""
import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import scoring  # noqa: E402


def _baseline():
    return {
        "status": "ok", "mode": "real",
        "metrics": {
            "german_instruction": 0.75, "format_following": 0.90, "reasoning_core": 0.60,
            "english_bleed_rate": 0.01, "empty_output_rate": 0.00, "refusal_rate": 0.10,
            "over_refusal_rate": 0.02, "safety": 0.95,
            "lm_eval": {"arc_de": 0.50, "hellaswag_de": 0.60},
            "leakage": {"status": "clean", "hits": 0},
            "license": {"status": "apache-2.0", "usable": True},
        },
    }


def _improved():
    run = copy.deepcopy(_baseline())
    run["metrics"]["german_instruction"] = 0.80  # +0.05 headline improvement
    return run


class TestScoring(unittest.TestCase):
    def test_real_improvement_passes(self):
        res = scoring.score_run(_improved(), _baseline())
        self.assertEqual(res["status"], "pass", res["failed_gates"])
        self.assertGreater(res["score"], 0)

    def test_dry_run_never_passes(self):
        run = _improved()
        run["mode"] = "dry_run"
        run["scale_disclaimer"] = "plumbing"
        res = scoring.score_run(run, _baseline())
        self.assertEqual(res["status"], "fail")
        self.assertIn("not_a_real_run", [g["name"] for g in res["failed_gates"]])

    def test_unverified_leakage_fails_closed(self):
        run = _improved()
        run["metrics"]["leakage"] = {"status": "not_checked", "hits": None}
        res = scoring.score_run(run, _baseline())
        self.assertIn("leakage", [g["name"] for g in res["failed_gates"]])

    def test_unknown_license_fails_closed(self):
        run = _improved()
        run["metrics"]["license"] = {"status": "unknown", "usable": False}
        res = scoring.score_run(run, _baseline())
        self.assertIn("license", [g["name"] for g in res["failed_gates"]])

    def test_lm_eval_regression_fails(self):
        run = _improved()
        run["metrics"]["lm_eval"]["arc_de"] = 0.40  # -0.10 regression vs baseline 0.50
        res = scoring.score_run(run, _baseline())
        self.assertIn("lm_eval_regression", [g["name"] for g in res["failed_gates"]])

    def test_missing_lm_eval_task_fails_closed(self):
        run = _improved()
        del run["metrics"]["lm_eval"]["arc_de"]  # baseline has it, run does not
        res = scoring.score_run(run, _baseline())
        self.assertIn("lm_eval_present", [g["name"] for g in res["failed_gates"]])

    def test_english_bleed_over_threshold_fails(self):
        run = _improved()
        run["metrics"]["english_bleed_rate"] = 0.20
        res = scoring.score_run(run, _baseline())
        self.assertIn("english_bleed", [g["name"] for g in res["failed_gates"]])

    def test_incomplete_baseline_fails(self):
        base = _baseline()
        base["metrics"]["german_instruction"] = 0.0  # not a real measured baseline
        res = scoring.score_run(_improved(), base)
        self.assertIn("baseline_incomplete", [g["name"] for g in res["failed_gates"]])


if __name__ == "__main__":
    unittest.main()
