import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from boldt_posttrain.data_pipeline import (
    DataError,
    deduplicate,
    leakage_filter,
    normalize_row,
    prepare,
    verify_data_manifest,
)
from boldt_posttrain.artifacts import atomic_write_json, sha256_file
from boldt_posttrain.evaluation import load_suite
from boldt_posttrain.policy import load_policy
from tests.tiny_model import build_tiny_model

FIXTURES = Path(__file__).parent / "fixtures/datasets"
SOURCE = {
    "dataset_id": "openeurollm/fixture",
    "revision": "a" * 40,
    "config": "default",
    "split": "train",
    "license": "Apache-2.0",
}


def rows(name: str):
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


def test_normalizes_real_file_fixtures_for_all_three_schemas():
    sft = normalize_row(rows("sft.jsonl")[0], SOURCE, "0")
    preference = normalize_row(rows("preference.jsonl")[0], SOURCE, "0")
    cpt = normalize_row(rows("cpt.jsonl")[0], SOURCE, "0")
    assert [sft["type"], preference["type"], cpt["type"]] == ["sft", "preference", "cpt"]
    assert all(len(item["content_id"]) == 64 for item in (sft, preference, cpt))


def test_unparseable_preference_row_is_rejected():
    with pytest.raises(DataError, match="distinct"):
        normalize_row(rows("preference.jsonl")[1], SOURCE, "1")


def test_exact_and_near_deduplication():
    first = normalize_row(rows("sft.jsonl")[0], SOURCE, "0")
    duplicate = normalize_row(rows("sft.jsonl")[0], SOURCE, "1")
    near_raw = rows("sft.jsonl")[0]
    near_raw["messages"][1]["content"] += "!"
    near = normalize_row(near_raw, SOURCE, "2")
    kept, stats = deduplicate([first, duplicate, near], 0.8)
    assert len(kept) == 1
    assert stats == {"exact_removed": 1, "near_removed": 1}


def test_exact_and_near_benchmark_leakage_is_removed():
    prompt = load_suite()[0]["prompt"]
    exact = normalize_row(
        {"prompt": prompt, "response": "Eine harmlose Antwort auf Deutsch."}, SOURCE, "0"
    )
    near = normalize_row(
        {"prompt": prompt + "!", "response": "Eine weitere Antwort."},
        SOURCE,
        "1",
    )
    clean, report = leakage_filter([exact, near], load_policy())
    assert clean == []
    assert report["status"] == "leak_detected"
    assert {match["type"] for hit in report["hits"] for match in hit["matches"]} >= {
        "exact",
        "near",
    }
    assert all("prompt" not in hit for hit in report["hits"])


class GermanLanguage:
    def __init__(self, policy):
        self.policy = policy

    def check(self, text: str) -> tuple[bool, float]:
        return True, 0.99


class RejectedLanguage(GermanLanguage):
    def check(self, text: str) -> tuple[bool, float]:
        return False, 0.01


def materialization_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    license_value: str = "apache-2.0",
    language_class=GermanLanguage,
):
    import boldt_posttrain.data_pipeline as pipeline
    import huggingface_hub
    from transformers import AutoTokenizer

    repository = tmp_path / "repo"
    outputs = repository / "outputs/posttrain"
    model_path = build_tiny_model(repository / "tiny")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    revision = "a" * 40

    class Hub:
        def dataset_info(self, dataset_id, revision: str):
            return SimpleNamespace(
                sha=revision,
                card_data={"license": license_value},
                gated=False,
                private=False,
                siblings=[],
            )

    monkeypatch.setattr(pipeline, "OUTPUTS", outputs)
    monkeypatch.setattr(pipeline, "LanguageIdentifier", language_class)
    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    config = json.loads((Path(__file__).parents[1] / "configs/posttrain/current.json").read_text())
    config["data"] = {
        "sources": [
            {
                "dataset_id": "openeurollm/fixture",
                "revision": revision,
                "config": "default",
                "split": "train",
                "schema": "sft",
            }
        ],
        "max_rows": 10,
    }
    config_path = repository / "experiment.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    return repository, outputs, config_path


def test_prepare_materializes_verified_trainable_file_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository, outputs, config_path = materialization_environment(tmp_path, monkeypatch)
    result = prepare(
        load_policy(),
        config_path,
        rows_provider=lambda source: iter(rows("sft.jsonl")),
    )
    assert result["status"] == "succeeded"
    manifest = verify_data_manifest(outputs / "data", load_policy(), repository_root=repository)
    assert manifest["status"] == "trainable"
    assert manifest["token_statistics"]["count"] == 2
    assert (outputs / "data" / result["run_id"] / "discovery.json").is_file()


def test_prepare_rejects_unknown_license_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, outputs, config_path = materialization_environment(
        tmp_path, monkeypatch, license_value="custom"
    )
    with pytest.raises(DataError, match="license"):
        prepare(load_policy(), config_path, rows_provider=lambda source: iter(rows("sft.jsonl")))
    assert not (outputs / "data/current.json").exists()


def test_prepare_rejects_wrong_language_and_empty_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, outputs, config_path = materialization_environment(
        tmp_path, monkeypatch, language_class=RejectedLanguage
    )
    with pytest.raises(DataError, match="no trainable rows"):
        prepare(load_policy(), config_path, rows_provider=lambda source: iter(rows("sft.jsonl")))
    assert not (outputs / "data/current.json").exists()


def test_prepare_rejects_exact_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, outputs, config_path = materialization_environment(tmp_path, monkeypatch)
    leaking = {
        "prompt": load_suite()[0]["prompt"],
        "response": "Eine deutsche Antwort mit ausreichend langem Inhalt.",
    }
    with pytest.raises(DataError, match="leakage"):
        prepare(load_policy(), config_path, rows_provider=lambda source: iter([leaking]))
    assert not (outputs / "data/current.json").exists()


def test_prepare_rejects_interrupted_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, outputs, config_path = materialization_environment(tmp_path, monkeypatch)

    def interrupted(source):
        yield rows("sft.jsonl")[0]
        raise OSError("fixture stream interrupted")

    with pytest.raises(DataError, match="stream interrupted"):
        prepare(load_policy(), config_path, rows_provider=interrupted)
    assert not (outputs / "data/current.json").exists()


def test_training_rejects_manifest_without_eval_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository, outputs, config_path = materialization_environment(tmp_path, monkeypatch)
    result = prepare(
        load_policy(), config_path, rows_provider=lambda source: iter(rows("sft.jsonl"))
    )
    run_dir = outputs / "data" / result["run_id"]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("eval_suite_hash")
    atomic_write_json(manifest_path, manifest)
    pointer_path = outputs / "data/current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(pointer_path, pointer)
    with pytest.raises(DataError, match="eval-suite fingerprint"):
        verify_data_manifest(outputs / "data", load_policy(), repository_root=repository)
