import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from boldt_posttrain.data_pipeline import (
    IntegrityError,
    FastTextLanguageIdentifier,
    _source_rows,
    _sampling_key,
    build_decontamination_corpus,
    canonical_json,
    discover_sources,
    distribution,
    make_selection_artifact,
    materialize_streaming,
    select_sources,
    normalize_row,
    sha256_bytes,
    stream_jsonl,
    verify_data_manifest,
    verify_selection_against_discovery,
)


class MaskingTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=False,
        tools=None,
    ):
        if tools is not None and not isinstance(tools, list):
            raise ValueError("bad tools")
        ids = []
        mask = []
        for message in messages:
            words = [message["role"], *message.get("content", "").split()]
            ids.extend(range(1, len(words) + 1))
            mask.extend([int(message["role"] == "assistant")] * len(words))
        if add_generation_prompt:
            ids.append(1)
            mask.append(0)
        result = {"input_ids": ids}
        if return_assistant_tokens_mask:
            result["assistant_masks"] = mask
        return result

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(str(text).split())))}


def source(rows, *, group="general", schema="sft"):
    return {
        "dataset": f"fixture-{group}",
        "revision": "fixed",
        "config": "default",
        "split": "train",
        "source_group": group,
        "schema": schema,
        "license": "apache-2.0",
        "rows": rows,
    }


def prepare_script_main():
    script = Path(__file__).resolve().parents[1] / "scripts/pt_prepare_openeurollm_de.py"
    spec = importlib.util.spec_from_file_location("pt_prepare_openeurollm_de_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def selected_ids(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        row["content_id"]
        for shard in manifest["shards"]
        for row in stream_jsonl(directory / shard["path"])
    }


def test_sampling_and_shards_are_independent_of_stream_order(tmp_path):
    rows = [{"instruction": f"Frage {i}", "answer": f"Antwort {i}"} for i in range(30)]
    first = materialize_streaming(
        [source(rows)], tmp_path / "first", max_rows_per_source=8, global_max_rows=8
    )
    second = materialize_streaming(
        [source(list(reversed(rows)))],
        tmp_path / "second",
        max_rows_per_source=8,
        global_max_rows=8,
    )
    assert selected_ids(tmp_path / "first") == selected_ids(tmp_path / "second")
    assert [shard["sha256"] for shard in first["shards"]] == [
        shard["sha256"] for shard in second["shards"]
    ]
    quality = json.loads((tmp_path / "first" / "quality_report.json").read_text())
    assert quality["sampled_per_source"] == [
        {
            "dataset": "fixture-general",
            "revision": "fixed",
            "config": "default",
            "split": "train",
            "schema": "sft",
            "rows_sampled": 8,
        }
    ]


def test_late_row_can_be_selected_and_sources_can_be_reordered(tmp_path):
    rows = [{"instruction": f"Frage {i}", "answer": f"Antwort {i}"} for i in range(20)]
    sampling_source = source(rows, group="left")
    normalized = [normalize_row(row, "sft", sampling_source) for row in rows]
    best = min(
        range(len(rows)),
        key=lambda index: _sampling_key(17, sampling_source, normalized[index]["content_id"]),
    )
    reordered = [row for index, row in enumerate(rows) if index != best] + [rows[best]]
    left = source(reordered, group="left")
    right = source([{"instruction": "Andere Frage", "answer": "Andere Antwort"}], group="right")
    first = materialize_streaming(
        [left, right], tmp_path / "one", max_rows_per_source=1, global_max_rows=2
    )
    second = materialize_streaming(
        [right, left], tmp_path / "two", max_rows_per_source=1, global_max_rows=2
    )
    assert normalized[best]["content_id"] in selected_ids(tmp_path / "one")
    assert first["artifact_hash"] == second["artifact_hash"]


def test_splits_are_disjoint_reproducible_and_have_expected_size(tmp_path):
    rows = [{"instruction": f"Frage {i}", "answer": f"Antwort {i}"} for i in range(20)]
    kwargs = {"validation_fraction": {"sft": 0.2}, "seed": 9}
    first = materialize_streaming([source(rows)], tmp_path / "first", **kwargs)
    second = materialize_streaming([source(list(reversed(rows)))], tmp_path / "second", **kwargs)
    assert first["split_statistics"]["sft"] == {"train": 16, "validation": 4}
    assert first["artifact_hash"] == second["artifact_hash"]
    train = {
        row["content_id"]
        for shard in first["shards"]
        if shard["split"] == "train"
        for row in stream_jsonl(tmp_path / "first" / shard["path"])
    }
    validation = {
        row["content_id"]
        for shard in first["shards"]
        if shard["split"] == "validation"
        for row in stream_jsonl(tmp_path / "first" / shard["path"])
    }
    assert train.isdisjoint(validation)


def test_conversational_preference_keeps_context_and_string_input_is_canonical():
    conversational = normalize_row(
        {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "human", "content": "Erste Frage"},
                {"role": "gpt", "content": "Erste Antwort"},
                {"role": "user", "content": "Zweite Frage"},
            ],
            "chosen": [{"role": "assistant", "content": "Gut"}],
            "rejected": [{"role": "assistant", "content": "Schlecht"}],
        },
        "sft",
        source([], schema="preference"),
    )
    assert conversational["schema"] == "preference"
    assert [message["role"] for message in conversational["prompt"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    strings = normalize_row(
        {"prompt": "Frage", "chosen": "Gut", "rejected": "Schlecht"},
        "preference",
        source([], schema="preference"),
    )
    assert strings["prompt"] == [{"role": "user", "content": "Frage"}]
    assert strings["chosen"] == [{"role": "assistant", "content": "Gut"}]
    full = normalize_row(
        {
            "chosen": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Frage"},
                {"role": "assistant", "content": "Gut"},
            ],
            "rejected": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Frage"},
                {"role": "assistant", "content": "Schlecht"},
            ],
        },
        "preference",
        source([], schema="preference"),
    )
    assert [message["role"] for message in full["prompt"]] == ["system", "user"]
    assert full["chosen"] == [{"role": "assistant", "content": "Gut"}]


def test_tools_are_preserved_and_bad_tools_are_rejected():
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    row = normalize_row(
        {
            "messages": [
                {"role": "user", "content": "Suche"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "Ergebnis", "tool_call_id": "1", "name": "lookup"},
                {"role": "assistant", "content": "Fertig"},
            ],
            "tools": [tool],
        },
        "sft",
        source([]),
    )
    assert row["tools"] == [tool]
    assert row["messages"][1]["tool_calls"][0]["id"] == "1"
    with pytest.raises(ValueError, match="tools"):
        normalize_row(
            {
                "messages": [
                    {"role": "user", "content": "X"},
                    {"role": "assistant", "content": "Y"},
                ],
                "tools": "lost",
            },
            "sft",
            source([]),
        )
    with pytest.raises(ValueError, match="legacy message field"):
        normalize_row(
            {
                "messages": [
                    {"role": "user", "content": "X", "functions": [{"name": "old"}]},
                    {"role": "assistant", "content": "Y"},
                ]
            },
            "sft",
            source([]),
        )


def test_remote_source_loading_preserves_revision_and_data_files(monkeypatch):
    calls = []

    def load_dataset(dataset, config, **kwargs):
        calls.append((dataset, config, kwargs))
        return [{"instruction": "Frage", "answer": "Antwort"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    rows = list(
        _source_rows(
            {
                "dataset": "org/pinned",
                "config": "default",
                "split": "train",
                "revision": "abc123",
                "data_files": ["de.jsonl"],
            }
        )
    )
    assert rows == [{"instruction": "Frage", "answer": "Antwort"}]
    assert calls == [
        (
            "org/pinned",
            "default",
            {
                "split": "train",
                "streaming": True,
                "revision": "abc123",
                "data_files": ["de.jsonl"],
            },
        )
    ]


def test_configured_discovery_and_selection_preserve_exact_source_pin():
    info = {
        "dataset": "allenai/Dolci-Instruct-SFT",
        "org": "allenai",
        "revision": "fixed",
        "configs": ["default"],
        "splits": ["train"],
        "schema": "sft",
        "license": "odc-by",
        "source_group": "dolci-de",
        "language_filter": "de",
        "samples": [
            {
                "messages": [
                    {"role": "user", "content": "Frage"},
                    {"role": "assistant", "content": "Antwort"},
                ]
            }
        ],
    }
    candidates = discover_sources(org="allenai", allowed_licenses=["odc-by"], dataset_infos=[info])
    assert candidates[0]["revision"] == "fixed"
    assert candidates[0]["language_filter"] == "de"
    selected = select_sources(
        candidates,
        allowed_org="allenai",
        allowed_licenses=["odc-by"],
        allowed_sources=[
            {
                "dataset": "allenai/Dolci-Instruct-SFT",
                "revision": "fixed",
                "license": "odc-by",
            }
        ],
    )
    assert selected == candidates
    assert not select_sources(
        candidates,
        allowed_org="allenai",
        allowed_licenses=["odc-by"],
        allowed_sources=[
            {
                "dataset": "allenai/Dolci-Instruct-SFT",
                "revision": "different",
                "license": "odc-by",
            }
        ],
    )


def test_selection_must_be_reproducible_from_discovery_and_policy():
    candidate = {
        "dataset": "allenai/Dolci-Instruct-SFT",
        "org": "allenai",
        "revision": "fixed",
        "config": "default",
        "split": "train",
        "schema": "sft",
        "license": "odc-by",
        "training_usable": True,
        "german_sample_ratio": 1.0,
        "row_count": 10,
    }
    discovery_body = {"run_id": "discovery-one", "candidates": [candidate]}
    discovery = {
        **discovery_body,
        "artifact_hash": sha256_bytes(canonical_json(discovery_body)),
    }
    allowed = [{"dataset": candidate["dataset"], "revision": "fixed", "license": "odc-by"}]
    selection = make_selection_artifact(
        "discovery-one",
        [candidate],
        allowed_org="allenai",
        allowed_licenses=["odc-by"],
        allowed_sources=allowed,
    )
    verify_selection_against_discovery(
        discovery,
        selection,
        allowed_org="allenai",
        allowed_licenses=["odc-by"],
        allowed_sources=allowed,
    )

    forged_body = {key: value for key, value in selection.items() if key != "artifact_hash"}
    forged_body["sources"] = [
        dict(candidate, dataset="unapproved/other", license="unknown", rows=[])
    ]
    forged = {**forged_body, "artifact_hash": sha256_bytes(canonical_json(forged_body))}
    with pytest.raises(IntegrityError, match="selected sources differ"):
        verify_selection_against_discovery(
            discovery,
            forged,
            allowed_org="allenai",
            allowed_licenses=["odc-by"],
            allowed_sources=allowed,
        )


def test_explicit_language_filter_runs_before_bounded_sampling(tmp_path):
    rows = [
        {"instruction": f"{'DE' if index >= 10 else 'EN'} Frage {index}", "answer": "Antwort"}
        for index in range(14)
    ]
    identifier = FastTextLanguageIdentifier(
        predictor=lambda text: ("de", 0.99) if "DE Frage" in text else ("en", 0.99)
    )
    configured = source(rows)
    configured["language_filter"] = "de"
    manifest = materialize_streaming(
        [configured],
        tmp_path / "data",
        language_id=identifier,
        min_german_confidence=0.8,
        max_rows_per_source=3,
        global_max_rows=3,
    )
    assert manifest["row_counts"]["rows_scanned"] == 14
    assert manifest["row_counts"]["language_rejected"] == 10
    assert manifest["row_counts"]["rows_sampled"] == 3
    assert manifest["row_counts"]["rows_trainable"] == 3


def test_language_filter_checks_substantive_assistant_outputs_separately(tmp_path):
    def predict(text):
        if "Deutsche Frage" in text or "deutsche Antwort" in text:
            return "de", 0.99
        return "en", 0.99

    configured = source(
        [
            {
                "instruction": "Deutsche Frage mit ausreichend langem Inhalt",
                "answer": "Das ist eine längere deutsche Antwort für diesen Test.",
            },
            {
                "instruction": "Deutsche Frage mit ausreichend langem Inhalt zwei",
                "answer": "This is a substantive English answer that must be rejected.",
            },
            {"instruction": "Deutsche Frage als Klassifikation", "answer": "POS"},
        ]
    )
    configured["language_filter"] = "de"
    manifest = materialize_streaming(
        [configured],
        tmp_path / "data",
        language_id=FastTextLanguageIdentifier(predictor=predict),
        min_german_confidence=0.8,
    )
    assert manifest["row_counts"]["rows_trainable"] == 2
    assert manifest["rejection_reasons"]["assistant_language"] == 1


def test_template_statistics_include_masks_percentiles_and_truncation(tmp_path):
    rows = [
        {"instruction": "eins zwei", "answer": "drei vier"},
        {"instruction": "eins", "answer": "drei"},
    ]
    manifest = materialize_streaming(
        [source(rows)],
        tmp_path / "data",
        tokenizer=MaskingTokenizer(),
        context_length=5,
    )
    stats = manifest["token_statistics"]["sft"]
    assert stats["assistant_tokens"]["count"] == 2
    assert stats["supervised_token_fraction"] > 0
    assert stats["truncation_fraction"] == 0.5
    assert distribution([1, 2, 3, 4])["p95"] == pytest.approx(3.85)


def test_tampered_validation_shard_is_detected(tmp_path):
    corpus = build_decontamination_corpus(
        [], tmp_path / "corpus.json", sources=[], policy_hash="policy"
    )
    directory = tmp_path / "data"
    manifest = materialize_streaming(
        [source([{"instruction": f"Frage {i}", "answer": f"Antwort {i}"} for i in range(5)])],
        directory,
        validation_fraction={"sft": 0.2},
        decontamination_corpus=corpus,
        policy_hash="policy",
    )
    validation = next(shard for shard in manifest["shards"] if shard["split"] == "validation")
    with (directory / validation["path"]).open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(IntegrityError, match="shard hash"):
        verify_data_manifest(directory / "manifest.json", expected_policy_hash="policy")


def test_failed_prepare_does_not_replace_existing_authoritative_artifacts(tmp_path):
    main = prepare_script_main()
    output = tmp_path / "data"
    output.mkdir()
    originals = {
        "manifest.json": '{"status":"trainable"}\n',
        "leakage_report.json": '{"status":"verified_clean"}\n',
        "quality_report.json": '{"status":"ok"}\n',
    }
    for name, content in originals.items():
        (output / name).write_text(content, encoding="utf-8")
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{}", encoding="utf-8")

    result = main(["--config", str(invalid_config), "--out", str(output), "--real"])

    assert result == 4
    assert {name: (output / name).read_text(encoding="utf-8") for name in originals} == originals


def test_dry_prepare_publishes_below_authoritative_directory(tmp_path):
    main = prepare_script_main()
    output = tmp_path / "data"
    output.mkdir()
    authoritative = output / "manifest.json"
    authoritative.write_text('{"status":"trainable"}\n', encoding="utf-8")
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{}", encoding="utf-8")

    assert main(["--config", str(invalid_config), "--out", str(output)]) == 4
    assert authoritative.read_text(encoding="utf-8") == '{"status":"trainable"}\n'
    assert (output / "dry-run/manifest.json").is_file()
