import json
import sys
import types
from datetime import datetime, timezone

import pytest

from boldt_posttrain.data_pipeline import (
    build_decontamination_corpus,
    decontaminate,
    IntegrityError,
    lm_eval_decontamination_records,
    materialize_streaming,
    verify_trainable_manifest,
)


def test_corpus_covers_context_answers_and_options(tmp_path):
    corpus = build_decontamination_corpus(
        [
            {
                "prompt": "Frage",
                "context": "Kontext",
                "expected": "Antwort",
                "options": ["A", "B"],
                "document": "Dokument",
            }
        ],
        tmp_path / "decontam.json",
        sources=[{"revision": "abc"}],
        policy_hash="policy",
    )
    values = {entry["canonical"] for entry in corpus["entries"]}
    assert values == {"Frage", "Kontext", "Antwort", "A", "B", "Dokument"}
    assert not decontaminate({"text": "Kontext"}, corpus)
    assert decontaminate({"text": "Ein unabhängiger Text enthält Kontext und Berlin."}, corpus)


def test_long_benchmark_prompt_is_detected_inside_a_wrapped_prompt(tmp_path):
    prompt = "Diese ausreichend lange geheime Prüfungsfrage darf niemals kopiert werden"
    corpus = build_decontamination_corpus(
        [{"prompt": prompt}], tmp_path / "decontam.json", sources=[], policy_hash="policy"
    )
    assert not decontaminate({"text": f"Anweisung: {prompt}\nAntwort:"}, corpus)


def test_materialization_rejects_overlap_and_training_validation_is_fail_closed(tmp_path):
    data = tmp_path / "data"
    corpus = build_decontamination_corpus(
        [{"prompt": "Geheime Prüfungsfrage"}],
        tmp_path / "corpus.json",
        sources=[{"source": "dev", "revision": "one"}],
        policy_hash="policy",
    )
    source = {
        "dataset": "fixture",
        "source_group": "general",
        "schema": "sft",
        "license": "apache-2.0",
        "rows": [
            {"instruction": "Geheime Prüfungsfrage", "answer": "verboten"},
            {"instruction": "Unabhängige Aufgabe", "answer": "erlaubt"},
        ],
    }
    manifest = materialize_streaming(
        [source],
        data,
        decontamination_corpus=corpus,
        policy_hash="policy",
    )
    assert manifest["row_counts"]["leakage_rejected"] == 1
    assert verify_trainable_manifest(data / "manifest.json", expected_policy_hash="policy")
    manifest["decontamination_hash"] = "stale"
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_trainable_manifest(data / "manifest.json", expected_policy_hash="policy")


def test_lm_eval_decontamination_uses_an_actually_readable_split(monkeypatch):
    class Task:
        has_validation_docs = has_test_docs = has_training_docs = lambda self: True

        def validation_docs(self):
            raise KeyError("validation")

        def test_docs(self):
            return [{"question": "Frage", "created": datetime(2026, 1, 1, tzinfo=timezone.utc)}]

        def training_docs(self):
            raise AssertionError("the test split should win")

        def doc_to_text(self, document):
            return document["question"]

        def doc_to_target(self, _document):
            return "Antwort"

    manager = types.SimpleNamespace(load_task_or_group=lambda _tasks: {"task": Task()})
    module = types.SimpleNamespace(TaskManager=lambda: manager)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", module)
    records = list(lm_eval_decontamination_records(["task"]))
    assert records[0]["prompt"] == "Frage"
    assert "2026-01-01" in records[0]["document"]
