"""Protected-surface classification of the integrity guard (pure stdlib unittest)."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_integrity():
    path = ROOT / "scripts" / "check_posttrain_integrity.py"
    spec = importlib.util.spec_from_file_location("check_posttrain_integrity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestIntegrity(unittest.TestCase):
    def setUp(self):
        self.mod = _load_integrity()
        self.globs = self.mod.load_globs()

    def test_editable_surface_passes(self):
        cls = self.mod.classify_paths(
            ["configs/posttrain/current.json",
             "configs/posttrain/experiments/foo.json",
             "docs/experiments/note.md"], self.globs)
        self.assertEqual(cls["protected"], [])
        self.assertEqual(len(cls["editable"]), 3)

    def test_protected_scripts_flagged(self):
        for p in ("scripts/pt_score.py", "scripts/pt_eval.py", "scripts/pt_promote.py",
                  "scripts/check_posttrain_integrity.py", "src/boldt_posttrain/scoring.py",
                  "CLAUDE.md", "AUTORESEARCH_POSTTRAIN.md"):
            res = self.mod.evaluate([p], globs=self.globs)
            self.assertEqual(res["status"], "fail", f"{p} should be protected")
            self.assertIn(p, res["violations"])

    def test_baseline_outputs_protected(self):
        res = self.mod.evaluate(["outputs/posttrain/baseline/summary.json"], globs=self.globs)
        self.assertEqual(res["status"], "fail")

    def test_unrelated_file_is_other_not_protected(self):
        res = self.mod.evaluate(["README.md"], globs=self.globs)
        self.assertEqual(res["status"], "pass")          # 'other', not protected
        res_strict = self.mod.evaluate(["README.md"], strict=True, globs=self.globs)
        self.assertEqual(res_strict["status"], "fail")   # strict flags anything non-editable


if __name__ == "__main__":
    unittest.main()
