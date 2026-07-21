"""Immutable artifact primitives, atomic publication, and hash-chained events."""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[a-z][a-z0-9-]*-\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8,16}$")
EXCLUDED_DIRECTORY_NAMES = {"__pycache__", ".cache"}
EXCLUDED_FILE_SUFFIXES = {".lock", ".tmp"}
SECRET_FLAGS = {"--token", "--api-key", "--password", "--secret", "--hf-token"}


class ArtifactError(ValueError):
    """An artifact is malformed, unsafe, missing, or hash-invalid."""


def _walk_finite(value: Any, location: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"{location} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArtifactError(f"{location} contains a non-string object key")
            _walk_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_finite(child, f"{location}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _walk_finite(value)
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if (
            path.is_file()
            and path.suffix not in EXCLUDED_FILE_SUFFIXES
            and not path.name.startswith(".tmp-")
        ):
            yield path


def directory_manifest(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    if not root.is_dir():
        raise ArtifactError(f"not a directory: {root}")
    return [
        {
            "path": file.relative_to(root).as_posix(),
            "size_bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in _included_files(root)
    ]


def sha256_directory(path: str | Path) -> str:
    return sha256_bytes(canonical_json_bytes(directory_manifest(path)))


def directory_size(path: str | Path) -> int:
    return sum(file.stat().st_size for file in _included_files(Path(path)))


def ensure_within_root(path: str | Path, root: str | Path) -> Path:
    root_path = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ArtifactError(f"path escapes allowed root {root_path}: {path}") from exc
    return candidate


def validate_label(label: str) -> str:
    normalized = unicodedata.normalize("NFC", label)
    if normalized != label or not label or label in {".", ".."}:
        raise ArtifactError("label is empty, dot-like, or not NFC-normalized")
    if "/" in label or "\\" in label or Path(label).is_absolute() or ".." in label:
        raise ArtifactError("label contains a path component")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", label):
        raise ArtifactError("label contains unsupported characters")
    return label


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    validator: Callable[[Path], None] | None = None,
    before_replace: Callable[[Path], None] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if validator:
            validator(temporary)
        if before_replace:
            before_replace(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> Path:
    payload = canonical_json_bytes(value) + b"\n"

    def validate(candidate: Path) -> None:
        loaded = json.loads(
            candidate.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ArtifactError(item)),
        )
        _walk_finite(loaded)

    return atomic_write_bytes(path, payload, validator=validate)


@contextlib.contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclasses.dataclass(frozen=True)
class ArtifactRef:
    path: str
    kind: str
    role: str
    sha256: str
    size_bytes: int
    media_type: str

    @classmethod
    def from_path(
        cls, path: str | Path, *, role: str, media_type: str, relative_to: str | Path | None = None
    ) -> "ArtifactRef":
        source = Path(path)
        if source.is_file():
            kind, digest, size = "file", sha256_file(source), source.stat().st_size
        elif source.is_dir():
            kind, digest, size = "directory", sha256_directory(source), directory_size(source)
        else:
            raise ArtifactError(f"artifact does not exist: {source}")
        stored = (
            source.resolve().relative_to(Path(relative_to).resolve()).as_posix()
            if relative_to
            else str(source)
        )
        return cls(stored, kind, role, digest, size, media_type)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        required = {field.name for field in dataclasses.fields(cls)}
        if set(value) != required:
            raise ArtifactError(f"ArtifactRef keys must be exactly {sorted(required)}")
        ref = cls(**dict(value))
        if (
            ref.kind not in {"file", "directory"}
            or not SHA256_RE.fullmatch(ref.sha256)
            or ref.size_bytes < 0
        ):
            raise ArtifactError("ArtifactRef contains invalid kind, hash, or size")
        return ref

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def verify_artifact_ref(
    value: Mapping[str, Any] | ArtifactRef, *, root: str | Path | None = None
) -> Path:
    ref = value if isinstance(value, ArtifactRef) else ArtifactRef.from_dict(value)
    path = ensure_within_root(ref.path, root) if root else Path(ref.path)
    if ref.kind == "file" and not path.is_file():
        raise ArtifactError(f"artifact file missing: {path}")
    if ref.kind == "directory" and not path.is_dir():
        raise ArtifactError(f"artifact directory missing: {path}")
    actual_hash = sha256_file(path) if ref.kind == "file" else sha256_directory(path)
    actual_size = path.stat().st_size if ref.kind == "file" else directory_size(path)
    if actual_hash != ref.sha256 or actual_size != ref.size_bytes:
        raise ArtifactError(f"artifact hash or size mismatch: {path}")
    return path


def new_run_id(run_type: str) -> str:
    prefix = re.sub(r"[^a-z0-9]+", "-", run_type.lower()).strip("-")
    if not prefix or not prefix[0].isalpha():
        raise ArtifactError("run type must begin with a letter")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}-{timestamp}-{secrets.token_hex(8)}"


RUN_CARD_FIELDS = {
    "schema_version",
    "run_id",
    "run_type",
    "mode",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "command",
    "git",
    "policy",
    "experiment",
    "inputs",
    "outputs",
    "model",
    "data",
    "parameters",
    "hardware",
    "environment",
    "parents",
    "compatibility_fingerprint",
    "error",
}
RUN_TYPES = {
    "data_discover",
    "data_prepare",
    "baseline",
    "train_sft",
    "train_cpt",
    "train_preference",
    "distill",
    "merge",
    "eval",
    "score",
    "promote",
}
RUN_STATUSES = {"succeeded", "failed", "budget_exhausted", "rejected", "promoted"}


def validate_run_card(card: Mapping[str, Any]) -> None:
    missing, unknown = RUN_CARD_FIELDS - set(card), set(card) - RUN_CARD_FIELDS
    if missing or unknown:
        raise ArtifactError(
            f"run card fields invalid; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    _walk_finite(card)
    if card["schema_version"] != 1 or not RUN_ID_RE.fullmatch(str(card["run_id"])):
        raise ArtifactError("run card schema_version or run_id is invalid")
    if (
        card["run_type"] not in RUN_TYPES
        or card["mode"] != "real"
        or card["status"] not in RUN_STATUSES
    ):
        raise ArtifactError("run card type, mode, or status is invalid")
    if not isinstance(card["command"], list) or not all(
        isinstance(item, str) for item in card["command"]
    ):
        raise ArtifactError("run card command must be an argument list")
    environment = card["environment"]
    event_head = environment.get("event_head") if isinstance(environment, Mapping) else None
    if (
        not isinstance(event_head, Mapping)
        or set(event_head) != {"sequence", "last_event_hash", "log_sha256"}
        or not isinstance(event_head["sequence"], int)
        or event_head["sequence"] < 1
        or not SHA256_RE.fullmatch(str(event_head["last_event_hash"]))
        or not SHA256_RE.fullmatch(str(event_head["log_sha256"]))
    ):
        raise ArtifactError("run card must reference one valid observed event head")
    for field in ("inputs", "outputs"):
        if not isinstance(card[field], list):
            raise ArtifactError(f"run card {field} must be an array")
        for ref in card[field]:
            ArtifactRef.from_dict(ref)


def redact_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for argument in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        key = argument.split("=", 1)[0].lower()
        if key in SECRET_FLAGS:
            if "=" in argument:
                redacted.append(f"{key}=<redacted>")
            else:
                redacted.append(argument)
                hide_next = True
            continue
        redacted.append(argument)
    return redacted


class EventLog:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.log_path = self.root / "events.jsonl"
        self.head_path = self.root / "events.head.json"
        self.lock_path = self.root / ".events.lock"

    def append(self, event_type: str, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with exclusive_lock(self.lock_path):
            head = self.validate() if self.log_path.exists() or self.head_path.exists() else None
            sequence = 1 if head is None else head["sequence"] + 1
            previous = None if head is None else head["last_event_hash"]
            event: dict[str, Any] = {
                "sequence": sequence,
                "previous_event_hash": previous,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "run_id": run_id,
                "payload": dict(payload),
            }
            event["event_hash"] = sha256_bytes(canonical_json_bytes(event))
            previous_bytes = self.log_path.read_bytes() if self.log_path.exists() else b""
            log_bytes = previous_bytes + canonical_json_bytes(event) + b"\n"
            atomic_write_bytes(self.log_path, log_bytes)
            new_head = {
                "schema_version": 1,
                "sequence": sequence,
                "last_event_hash": event["event_hash"],
                "log_sha256": sha256_bytes(log_bytes),
            }
            atomic_write_json(self.head_path, new_head)
            return {**new_head, "event": event}

    def validate(self) -> dict[str, Any]:
        if not self.log_path.is_file() or not self.head_path.is_file():
            raise ArtifactError("event log and head must either both exist or both be absent")
        try:
            head = json.loads(self.head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"event head is unreadable: {exc}") from exc
        if (
            set(head) != {"schema_version", "sequence", "last_event_hash", "log_sha256"}
            or head["schema_version"] != 1
        ):
            raise ArtifactError("event head schema is invalid")
        log_bytes = self.log_path.read_bytes()
        if sha256_bytes(log_bytes) != head["log_sha256"]:
            raise ArtifactError("event log hash does not match head")
        previous = None
        count = 0
        for count, raw in enumerate(log_bytes.splitlines(), start=1):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"event {count} is invalid JSON") from exc
            event_hash = event.pop("event_hash", None)
            if event.get("sequence") != count or event.get("previous_event_hash") != previous:
                raise ArtifactError(f"event chain broken at sequence {count}")
            if event_hash != sha256_bytes(canonical_json_bytes(event)):
                raise ArtifactError(f"event hash mismatch at sequence {count}")
            previous = event_hash
        if count != head["sequence"] or previous != head["last_event_hash"]:
            raise ArtifactError("event log was truncated, extended, or exchanged")
        return head
