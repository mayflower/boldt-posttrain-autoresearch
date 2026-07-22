from datetime import datetime, timezone
from types import SimpleNamespace

from boldt_posttrain.data_pipeline import (
    LanguageIdentifier,
    classify_schema,
    discover,
    normalize_license,
)
from boldt_posttrain.policy import load_policy


def test_license_normalization_fails_closed():
    assert normalize_license("apache-2.0") == "Apache-2.0"
    assert normalize_license("custom") is None
    assert normalize_license(None) is None


def test_schema_discovery_only_accepts_explicit_forms():
    assert classify_schema({"prompt": "a", "response": "b"}) == "sft"
    assert classify_schema({"prompt": "a", "chosen": "b", "rejected": "c"}) == "preference"
    assert classify_schema({"text": "a"}) == "cpt"
    assert classify_schema({"foo": "bar"}) is None


def test_checked_in_language_model_and_hash_are_real():
    identifier = LanguageIdentifier(load_policy())
    german, confidence = identifier.check(
        "Dies ist ein längerer deutscher Beispielsatz über Sprache und zuverlässige Datenprüfung."
    )
    assert german is True
    assert confidence >= load_policy().document["data"]["min_german_confidence"]


class GermanLanguage:
    def __init__(self, policy):
        self.policy = policy

    def check(self, text: str) -> tuple[bool, float]:
        return bool(text.strip()), 0.99


def test_discovery_uses_exact_hub_revision_and_captures_split_evidence(monkeypatch):
    import boldt_posttrain.data_pipeline as pipeline
    import datasets

    revision = "a" * 40

    class Api:
        def list_datasets(self, *, author: str, full: bool):
            assert author == "openeurollm"
            assert full is True
            return [SimpleNamespace(id="openeurollm/fixture", sha=revision)]

        def dataset_info(self, dataset_id: str, *, revision: str, files_metadata: bool):
            assert dataset_id == "openeurollm/fixture"
            assert files_metadata is True
            return SimpleNamespace(
                sha=revision,
                card_data={
                    "license": "apache-2.0",
                    "dataset_info": {
                        "config_name": "default",
                        "splits": {"train": {"num_examples": 2}},
                    },
                },
                siblings=[],
                gated=False,
                private=False,
            )

    monkeypatch.setattr(pipeline, "LanguageIdentifier", GermanLanguage)
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda *args, **kwargs: ["default"])
    monkeypatch.setattr(datasets, "get_dataset_split_names", lambda *args, **kwargs: ["train"])
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: iter(
            [
                {"prompt": "Warum fällt Regen?", "response": "Wegen der Schwerkraft."},
                {
                    "prompt": "Nenne ein Buch.",
                    "response": "Ein Roman ist passend.",
                    "created_at": datetime(2026, 7, 21, tzinfo=timezone.utc),
                },
            ]
        ),
    )
    document = discover(load_policy(), api=Api())
    candidate = document["candidates"][0]
    assert candidate["dataset_revision_sha"] == revision
    assert candidate["row_estimate"] == 2
    assert candidate["normalized_spdx_license"] == "Apache-2.0"
    assert candidate["schema_classification"] == "sft"
    assert candidate["language_evidence"]["german"] == 2
    assert candidate["training_usable"] is True


def test_discovery_rejects_remote_code_without_executing_stream(monkeypatch):
    import boldt_posttrain.data_pipeline as pipeline
    import datasets

    revision = "b" * 40

    class Api:
        def list_datasets(self, *, author: str, full: bool):
            return [SimpleNamespace(id="openeurollm/scripted", sha=revision)]

        def dataset_info(self, dataset_id: str, *, revision: str, files_metadata: bool):
            return SimpleNamespace(
                sha=revision,
                card_data={"license": "apache-2.0"},
                siblings=[SimpleNamespace(rfilename="dataset.py")],
                gated=False,
                private=False,
            )

    monkeypatch.setattr(pipeline, "LanguageIdentifier", GermanLanguage)
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda *args, **kwargs: ["default"])
    monkeypatch.setattr(datasets, "get_dataset_split_names", lambda *args, **kwargs: ["train"])
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stream executed")),
    )
    candidate = discover(load_policy(), api=Api())["candidates"][0]
    assert candidate["training_usable"] is False
    assert "remote_code_required" in candidate["rejection_reasons"]
