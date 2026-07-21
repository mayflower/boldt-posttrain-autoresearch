import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from boldt_posttrain.artifacts import (
    ArtifactError,
    EventLog,
    RUN_ID_RE,
    atomic_write_bytes,
    ensure_within_root,
    new_run_id,
    sha256_directory,
)


def test_atomic_write_preserves_destination_on_pre_publish_exception(tmp_path: Path):
    destination = tmp_path / "value.json"
    destination.write_text("old")

    def fail(_: Path) -> None:
        raise RuntimeError("stop before replace")

    with pytest.raises(RuntimeError):
        atomic_write_bytes(destination, b"new", before_replace=fail)
    assert destination.read_text() == "old"
    assert not list(tmp_path.glob(".tmp-*"))


def test_run_ids_are_unique_under_concurrent_creation():
    with ThreadPoolExecutor(max_workers=16) as executor:
        run_ids = list(executor.map(lambda _: new_run_id("train-sft"), range(1000)))
    assert len(run_ids) == len(set(run_ids))
    assert all(RUN_ID_RE.fullmatch(run_id) for run_id in run_ids)


def test_directory_hash_is_stable_and_content_sensitive(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b").write_text("two")
    (first / "a").write_text("one")
    (second / "a").write_text("one")
    (second / "b").write_text("two")
    assert sha256_directory(first) == sha256_directory(second)
    (second / "b").write_text("changed")
    assert sha256_directory(first) != sha256_directory(second)


def test_user_labels_cannot_escape_output_root(tmp_path: Path):
    assert ensure_within_root("child/file", tmp_path) == tmp_path / "child/file"
    with pytest.raises(ArtifactError, match="escapes"):
        ensure_within_root("../outside", tmp_path)


def test_event_hash_chain_detects_modification_and_truncation(tmp_path: Path):
    events = EventLog(tmp_path)
    events.append("run_started", new_run_id("eval"), {})
    events.append("run_finished", new_run_id("eval"), {"status": "succeeded"})
    original = events.log_path.read_bytes()
    lines = original.splitlines()
    modified = json.loads(lines[0])
    modified["payload"] = {"tampered": True}
    lines[0] = json.dumps(modified).encode()
    events.log_path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ArtifactError):
        events.validate()
    events.log_path.write_bytes(original.splitlines(keepends=True)[0])
    with pytest.raises(ArtifactError):
        events.validate()
