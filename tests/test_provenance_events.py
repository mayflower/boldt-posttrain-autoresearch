from boldt_posttrain.provenance import append_event, validate_event_chain


def test_event_chain_detects_tampering(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, {"event": "one"})
    append_event(path, {"event": "two"})
    assert validate_event_chain(path) == []
    path.write_text(path.read_text().replace('"two"', '"tampered"'))
    assert validate_event_chain(path)
