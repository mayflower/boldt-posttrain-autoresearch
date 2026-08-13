from boldt_posttrain.data_pipeline import materialize_streaming, verify_hashed_artifact


def test_bounded_shards_and_deterministic_manifest(tmp_path):
    rows = [{"instruction": f"Frage {i}", "answer": f"Antwort {i}"} for i in range(7)]
    source = {
        "dataset": "fixture",
        "source_group": "general",
        "schema": "sft",
        "license": "apache-2.0",
        "rows": rows,
    }
    first = materialize_streaming([source], tmp_path / "first", max_rows=2)
    second = materialize_streaming([source], tmp_path / "second", max_rows=2)
    assert len(first["shards"]) == 4
    assert first["artifact_hash"] == second["artifact_hash"]
    assert verify_hashed_artifact(first)
    assert not (tmp_path / "first/near-dedup.sqlite3").exists()
    for shard in first["shards"]:
        assert len((tmp_path / "first" / shard["path"]).read_text().splitlines()) <= 2
