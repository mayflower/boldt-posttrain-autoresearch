"""pt_loop deterministic iteration: dry/missing-baseline runs are never promotable (unittest)."""
import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_loop():
    path = ROOT / "scripts" / "pt_loop.py"
    spec = importlib.util.spec_from_file_location("pt_loop", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv):
    mod = _load_loop()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(argv)
    return rc, json.loads(buf.getvalue())


class TestLoop(unittest.TestCase):
    def test_dry_run_is_not_promotable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            rc, verdict = _run(["--candidate", "baseline-seed", "--label", "t-dry",
                                "--evals-out", str(tmp / "evals"),
                                "--results", str(tmp / "results.tsv"),
                                "--baseline", str(tmp / "no-baseline.json"), "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertFalse(verdict["promotable"])
        self.assertEqual(verdict["mode"], "dry_run")
        # the eval ran and a results row was logged even though it is not promotable
        self.assertEqual(verdict["eval_status"], "ok")

    def test_missing_baseline_skips_score_and_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            rc, verdict = _run(["--label", "t-nob", "--evals-out", str(tmp / "evals"),
                                "--results", str(tmp / "results.tsv"),
                                "--baseline", str(tmp / "missing.json"), "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertIsNone(verdict["score_status"])      # score skipped (no baseline)
        self.assertFalse(verdict["baseline_present"])
        self.assertFalse(verdict["promotable"])


if __name__ == "__main__":
    unittest.main()
