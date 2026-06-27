"""Config inheritance + validation (pure stdlib unittest)."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain import config as cfgmod  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_current_extends_base(self):
        cfg = cfgmod.resolve_config(cfgmod.DEFAULT_CONFIG)
        # current.json declares extends -> base.json keys must be present after the merge.
        self.assertEqual(cfg.get("_extends"), "configs/posttrain/base.json")
        self.assertIn("paths", cfg)        # from base.json
        self.assertIn("integrity", cfg)    # from base.json
        self.assertIn("training", cfg)     # from current.json

    def test_resolved_current_is_valid(self):
        cfg = cfgmod.resolve_config(cfgmod.DEFAULT_CONFIG)
        self.assertEqual(cfgmod.validate_config_dict(cfg), [])

    def test_deep_merge_overrides_nested(self):
        base = {"training": {"lr": 1, "method": "qlora"}, "x": 1}
        overlay = {"training": {"lr": 2}, "y": 2}
        merged = cfgmod.deep_merge(base, overlay)
        self.assertEqual(merged["training"], {"lr": 2, "method": "qlora"})
        self.assertEqual(merged["x"], 1)
        self.assertEqual(merged["y"], 2)

    def test_validation_flags_missing_blocks(self):
        errors = cfgmod.validate_config_dict({"training": {}})
        self.assertTrue(any("base_model" in e for e in errors))
        self.assertTrue(any("data" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
