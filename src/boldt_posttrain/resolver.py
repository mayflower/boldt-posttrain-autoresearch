"""Exact, hash-verified model and candidate resolution."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    ArtifactError,
    ArtifactRef,
    EventLog,
    RUN_ID_RE,
    atomic_write_json,
    sha256_file,
    validate_label,
    validate_run_card,
    verify_artifact_ref,
)
from .policy import Policy

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "posttrain"
HUB_REF_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)@(?P<revision>[0-9a-f]{40})$"
)


class ResolutionError(RuntimeError):
    """A requested model does not resolve to one exact verified object."""


def load_tokenizer(
    model_source: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool | None = None,
):
    from transformers import AutoTokenizer

    source = str(model_source)
    local = Path(source).is_absolute() if local_files_only is None else local_files_only
    return AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        local_files_only=local,
        extra_special_tokens={},
    )


@dataclass(frozen=True)
class ResolvedModelRef:
    kind: str
    requested: str
    base_model: dict[str, str]
    artifact: dict[str, Any] | None
    tokenizer_sha256: str
    chat_template_sha256: str
    model_config_sha256: str
    architecture: str
    source_run_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ResolutionError(f"non-finite JSON: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResolutionError(f"expected JSON object at {path}")
    return value


def _verified_event_runs(outputs_root: Path) -> set[str]:
    events = EventLog(outputs_root)
    events.validate()
    successful: set[str] = set()
    for raw in events.log_path.read_bytes().splitlines():
        event = json.loads(raw)
        if (
            event.get("event_type") == "run_finished"
            and event.get("payload", {}).get("status") == "succeeded"
        ):
            run_id = str(event.get("run_id"))
            run_card = outputs_root / "runs" / run_id / "run_card.json"
            if run_card.is_file() and event["payload"].get("run_card_sha256") == sha256_file(
                run_card
            ):
                successful.add(run_id)
    return successful


def _resolve_pointer(requested: str, outputs_root: Path) -> str:
    if RUN_ID_RE.fullmatch(requested):
        return requested
    validate_label(requested)
    pointer_path = outputs_root / "registry" / "pointers" / f"{requested}.json"
    pointer = _load_json(pointer_path)
    if (
        set(pointer) != {"schema_version", "pointer_id", "run_id", "run_card_sha256"}
        or pointer.get("schema_version") != 1
    ):
        raise ResolutionError(f"pointer schema is invalid: {pointer_path}")
    if pointer.get("pointer_id") != requested or not RUN_ID_RE.fullmatch(
        str(pointer.get("run_id"))
    ):
        raise ResolutionError(f"pointer identity is invalid: {pointer_path}")
    run_card = outputs_root / "runs" / pointer["run_id"] / "run_card.json"
    if not run_card.is_file() or sha256_file(run_card) != pointer["run_card_sha256"]:
        raise ResolutionError(f"pointer run-card hash mismatch: {pointer_path}")
    return str(pointer["run_id"])


def _checkpoint_output(card: Mapping[str, Any]) -> ArtifactRef:
    matches: list[ArtifactRef] = []
    for value in card["outputs"]:
        ref = ArtifactRef.from_dict(value)
        if ref.role in {"adapter_checkpoint", "full_checkpoint", "merged_checkpoint"}:
            matches.append(ref)
    if len(matches) != 1:
        raise ResolutionError("candidate run card must reference exactly one checkpoint output")
    return matches[0]


def resolve_candidate(
    requested: str, policy: Policy, *, outputs_root: Path = OUTPUTS
) -> ResolvedModelRef:
    try:
        run_id = _resolve_pointer(requested, outputs_root)
    except (ArtifactError, ResolutionError, OSError) as exc:
        raise ResolutionError(f"unknown or invalid candidate {requested!r}: {exc}") from exc
    run_card_path = outputs_root / "runs" / run_id / "run_card.json"
    card = _load_json(run_card_path)
    try:
        validate_run_card(card)
    except ArtifactError as exc:
        raise ResolutionError(f"candidate run card is invalid: {exc}") from exc
    if card["mode"] != "real" or card["status"] != "succeeded":
        raise ResolutionError("candidate is not a successful real run")
    if run_id not in _verified_event_runs(outputs_root):
        raise ResolutionError("candidate has no successful event-chain record")
    checkpoint = _checkpoint_output(card)
    try:
        verify_artifact_ref(checkpoint, root=outputs_root.parents[1])
    except ArtifactError as exc:
        raise ResolutionError(f"candidate checkpoint verification failed: {exc}") from exc
    model = card["model"]
    if not isinstance(model, dict):
        raise ResolutionError("candidate model metadata is missing")
    base_model = model.get("base_model")
    expected_base = {
        "repo_id": policy.seed_model["repo_id"],
        "revision": policy.seed_model["revision"],
    }
    if base_model != expected_base:
        raise ResolutionError("candidate base model does not match the protected seed commit")
    required = ("tokenizer_sha256", "chat_template_sha256", "model_config_sha256", "architecture")
    if any(not isinstance(model.get(key), str) for key in required):
        raise ResolutionError("candidate compatibility fingerprints are incomplete")
    kind_by_role = {
        "adapter_checkpoint": "peft_adapter",
        "full_checkpoint": "local_full_checkpoint",
        "merged_checkpoint": "merged_checkpoint",
    }
    return ResolvedModelRef(
        kind=kind_by_role[checkpoint.role],
        requested=requested,
        base_model=base_model,
        artifact=checkpoint.to_dict(),
        tokenizer_sha256=model["tokenizer_sha256"],
        chat_template_sha256=model["chat_template_sha256"],
        model_config_sha256=model["model_config_sha256"],
        architecture=model["architecture"],
        source_run_id=run_id,
    )


def _fetch(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    except (OSError, urllib.error.HTTPError) as exc:
        raise ResolutionError(f"Hub request failed for {url}: {exc}") from exc


def resolve_hub_model(requested: str, policy: Policy) -> ResolvedModelRef:
    match = HUB_REF_RE.fullmatch(requested)
    if not match:
        if requested == policy.seed_model["repo_id"]:
            repo_id, revision = requested, policy.seed_model["revision"]
        else:
            raise ResolutionError("Hub models must use repo_id@40-character-commit")
    else:
        repo_id, revision = match.group("repo"), match.group("revision")
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    info = json.loads(
        _fetch(f"https://huggingface.co/api/models/{quoted_repo}/revision/{revision}")
    )
    if info.get("sha") != revision:
        raise ResolutionError("Hub returned a different model revision")
    files: dict[str, bytes] = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        files[name] = _fetch(f"https://huggingface.co/{quoted_repo}/resolve/{revision}/{name}")
    config = json.loads(files["config.json"])
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ResolutionError("Hub model must declare exactly one architecture")
    fingerprints = {
        name: __import__("hashlib").sha256(content).hexdigest() for name, content in files.items()
    }
    return ResolvedModelRef(
        kind="hub_model",
        requested=requested,
        base_model={"repo_id": repo_id, "revision": revision},
        artifact=None,
        tokenizer_sha256=fingerprints["tokenizer.json"],
        chat_template_sha256=fingerprints["chat_template.jinja"],
        model_config_sha256=fingerprints["config.json"],
        architecture=architectures[0],
        source_run_id=None,
    )


def resolve_local_model(
    requested: str, policy: Policy, *, allowed_roots: Sequence[Path]
) -> ResolvedModelRef:
    source = Path(requested).resolve()
    if not any(
        source == root.resolve() or source.is_relative_to(root.resolve()) for root in allowed_roots
    ):
        raise ResolutionError("local model is outside explicitly allowed read-only roots")
    if not source.is_dir():
        raise ResolutionError(f"local model directory does not exist: {source}")
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    if any(not (source / name).is_file() for name in required):
        raise ResolutionError("local full checkpoint lacks config/tokenizer/template files")
    config = _load_json(source / "config.json")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ResolutionError("local model must declare exactly one architecture")
    relative_root = ROOT if source.is_relative_to(ROOT) else None
    artifact = ArtifactRef.from_path(
        source,
        role="full_checkpoint",
        media_type="application/vnd.boldt.transformers-checkpoint",
        relative_to=relative_root,
    )
    return ResolvedModelRef(
        kind="local_full_checkpoint",
        requested=requested,
        base_model={
            "repo_id": policy.seed_model["repo_id"],
            "revision": policy.seed_model["revision"],
        },
        artifact=artifact.to_dict(),
        tokenizer_sha256=sha256_file(source / "tokenizer.json"),
        chat_template_sha256=sha256_file(source / "chat_template.jinja"),
        model_config_sha256=sha256_file(source / "config.json"),
        architecture=architectures[0],
        source_run_id=None,
    )


def resolve_model(
    *,
    policy: Policy,
    candidate: str | None = None,
    model: str | None = None,
    outputs_root: Path = OUTPUTS,
    external_roots: Sequence[Path] = (),
) -> ResolvedModelRef:
    if (candidate is None) == (model is None):
        raise ResolutionError("specify exactly one of candidate or model")
    if candidate is not None:
        return resolve_candidate(candidate, policy, outputs_root=outputs_root)
    assert model is not None
    if Path(model).is_absolute() or model.startswith("."):
        return resolve_local_model(
            model, policy, allowed_roots=(ROOT / "outputs/posttrain/checkpoints", *external_roots)
        )
    return resolve_hub_model(model, policy)


class CandidateRegistry:
    """A derived index; run cards and events remain the source of truth."""

    def __init__(self, outputs_root: Path = OUTPUTS):
        self.outputs_root = outputs_root
        self.path = outputs_root / "registry" / "current.json"

    def rebuild(self, policy: Policy) -> dict[str, Any]:
        runs_dir = self.outputs_root / "runs"
        candidates: dict[str, Any] = {}
        if runs_dir.exists():
            successful_events = _verified_event_runs(self.outputs_root)
            for run_card_path in sorted(runs_dir.glob("*/run_card.json")):
                run_id = run_card_path.parent.name
                if run_id not in successful_events:
                    continue
                try:
                    resolved = resolve_candidate(run_id, policy, outputs_root=self.outputs_root)
                except ResolutionError:
                    continue
                candidates[run_id] = {
                    "run_card_sha256": sha256_file(run_card_path),
                    "model": resolved.to_dict(),
                }
        document = {"schema_version": 1, "candidates": candidates}
        atomic_write_json(self.path, document)
        return document
