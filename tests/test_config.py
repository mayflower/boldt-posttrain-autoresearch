import json
from pathlib import Path

import pytest

from boldt_posttrain import config


def test_current_is_strict_and_valid():
    loaded = config.load_experiment()
    assert loaded.document["experiment"]["name"] == "current"
    assert "base_model" not in loaded.document["training"]


@pytest.mark.parametrize("key", ["threshold", "promotion", "base_model_revision", "eval_task"])
def test_experiment_cannot_override_policy_thresholds(tmp_path: Path, key: str):
    document = json.loads(config.DEFAULT_CONFIG.read_text())
    document[key] = {}
    path = tmp_path / "attack.json"
    path.write_text(json.dumps(document))
    with pytest.raises(config.ConfigError, match="protected policy key"):
        config.load_experiment(path)


def test_unknown_key_is_rejected(tmp_path: Path):
    document = json.loads(config.DEFAULT_CONFIG.read_text())
    document["training"]["surprise"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(document))
    with pytest.raises(config.ConfigError, match="unknown key"):
        config.load_experiment(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_numbers_are_rejected(tmp_path: Path, constant: str):
    text = config.DEFAULT_CONFIG.read_text().replace("0.00001", constant, 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(text)
    with pytest.raises(config.ConfigError, match="non-finite"):
        config.load_experiment(path)
