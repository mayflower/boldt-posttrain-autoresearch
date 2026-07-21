"""Atomic promotion and verified frontier history."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import provenance
from .artifacts import (
    ArtifactError,
    ArtifactRef,
    EventLog,
    atomic_write_json,
    canonical_json_bytes,
    exclusive_lock,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
    verify_artifact_ref,
)
from .evaluation import suite_hash
from .policy import Policy, load_policy
from .resolver import OUTPUTS, resolve_candidate
from .scoring import ScoringError, load_candidate_score

ROOT = Path(__file__).resolve().parents[2]


class FrontierError(RuntimeError):
    """Promotion evidence, integrity, or compare-and-swap validation failed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                FrontierError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrontierError(f"expected JSON object at {path}")
    return value


def _integrity_check(base_ref: str, repository_root: Path, policy: Policy) -> dict[str, Any]:
    script = repository_root / "scripts/check_posttrain_integrity.py"
    if not script.is_file():
        raise FrontierError("integrity checker is missing from the repository")
    spec = importlib.util.spec_from_file_location("promotion_integrity", script)
    if spec is None or spec.loader is None:
        raise FrontierError("integrity checker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.check(base_ref=base_ref, root=repository_root, policy_path=policy.path)
    if result.get("status") != "pass":
        raise FrontierError(f"integrity check failed: {result}")
    return result


def current_frontier_hash(frontier_root: Path) -> str | None:
    path = frontier_root / "current.json"
    return sha256_file(path) if path.is_file() else None


def _published_ref(source: Path, destination: Path, repository_root: Path) -> ArtifactRef:
    measured = ArtifactRef.from_path(
        source,
        role="promotion_verdict",
        media_type="application/vnd.boldt.promotion+json",
    )
    try:
        stored = destination.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        stored = str(destination.resolve())
    return ArtifactRef(
        stored,
        measured.kind,
        measured.role,
        measured.sha256,
        measured.size_bytes,
        measured.media_type,
    )


def _event_anchor(outputs_root: Path, promotion_id: str, run_card_sha256: str) -> dict[str, Any]:
    EventLog(outputs_root).validate()
    for line in (outputs_root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if (
            event.get("event_type") == "candidate_promoted"
            and event.get("run_id") == promotion_id
            and event.get("payload", {}).get("run_card_sha256") == run_card_sha256
        ):
            return event
    raise FrontierError("promotion run card is not anchored by candidate_promoted event")


def _verify_head_prefix(head: Mapping[str, Any], outputs_root: Path) -> None:
    if set(head) != {"sequence", "last_event_hash", "log_sha256"}:
        raise FrontierError("frontier event head schema is invalid")
    lines = (outputs_root / "events.jsonl").read_bytes().splitlines(keepends=True)
    sequence = head["sequence"]
    if not isinstance(sequence, int) or sequence < 1 or sequence > len(lines):
        raise FrontierError("frontier event head sequence is invalid")
    prefix = b"".join(lines[:sequence])
    event = json.loads(lines[sequence - 1])
    if (
        sha256_bytes(prefix) != head["log_sha256"]
        or event.get("event_hash") != head["last_event_hash"]
    ):
        raise FrontierError("frontier event head is not a current-chain prefix")


def verify_frontier(
    *,
    policy: Policy,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
) -> dict[str, Any] | None:
    current_path = outputs_root / "frontier/current.json"
    if not current_path.is_file():
        return None
    current = _load_json(current_path)
    required = {
        "schema_version",
        "promotion_id",
        "candidate_run_id",
        "score_run_id",
        "score",
        "policy_sha256",
        "suite_hash",
        "previous_frontier_sha256",
        "history_sha256",
        "run_card_sha256",
        "event_head",
    }
    if set(current) != required or current["schema_version"] != 1:
        raise FrontierError("frontier current pointer schema is invalid")
    if (
        current["policy_sha256"] != sha256_file(policy.path)
        or current["suite_hash"] != suite_hash()
    ):
        raise FrontierError("frontier policy or suite hash is stale")
    history_path = outputs_root / "frontier/history" / f"{current['promotion_id']}.json"
    run_card_path = outputs_root / "runs" / current["promotion_id"] / "run_card.json"
    if (
        sha256_file(history_path) != current["history_sha256"]
        or sha256_file(run_card_path) != current["run_card_sha256"]
    ):
        raise FrontierError("frontier history or run-card hash mismatch")
    history, card = _load_json(history_path), _load_json(run_card_path)
    validate_run_card(card)
    if (
        card["run_type"] != "promote"
        or card["status"] != "promoted"
        or history.get("status") != "promoted"
        or history.get("candidate_run_id") != current["candidate_run_id"]
        or history.get("score_run_id") != current["score_run_id"]
        or history.get("score") != current["score"]
    ):
        raise FrontierError("frontier history and run card are inconsistent")
    refs = [ArtifactRef.from_dict(item) for item in card["outputs"]]
    if (
        len(refs) != 1
        or verify_artifact_ref(refs[0], root=repository_root) != history_path.resolve()
    ):
        raise FrontierError("promotion run card does not hash its immutable verdict")
    _event_anchor(outputs_root, current["promotion_id"], current["run_card_sha256"])
    _verify_head_prefix(current["event_head"], outputs_root)
    artifact, _ = load_candidate_score(
        current["candidate_run_id"],
        policy,
        outputs_root=outputs_root,
        repository_root=repository_root,
    )
    if (
        artifact["run_id"] != current["score_run_id"]
        or artifact["status"] != "passed"
        or artifact["score"] != current["score"]
    ):
        raise FrontierError("frontier no longer matches its verified passing score")
    resolve_candidate(current["candidate_run_id"], policy, outputs_root=outputs_root)
    return current


def promote_candidate(
    candidate_run_id: str,
    *,
    base_ref: str,
    expected_current_sha256: str | None,
    policy: Policy,
    outputs_root: Path = OUTPUTS,
    repository_root: Path = ROOT,
    integrity_checker: Callable[[str, Path, Policy], dict[str, Any]] = _integrity_check,
) -> dict[str, Any]:
    started = time.monotonic()
    artifact, score_card = load_candidate_score(
        candidate_run_id,
        policy,
        outputs_root=outputs_root,
        repository_root=repository_root,
    )
    promotion = policy.document["scoring"]["promotion"]
    if (
        artifact["status"] != "passed"
        or not all(artifact["gates"].values())
        or artifact["score"] <= promotion["min_score"]
        or artifact["deltas"]["german_instruction"] < promotion["german_instruction_min_delta"]
    ):
        raise FrontierError("candidate score does not satisfy protected promotion gates")
    integrity = integrity_checker(base_ref, repository_root, policy)
    if integrity.get("status") != "pass":
        raise FrontierError("integrity checker did not return pass")
    frontier_root = outputs_root / "frontier"
    with exclusive_lock(frontier_root / ".frontier.lock"):
        actual_current_hash = current_frontier_hash(frontier_root)
        if actual_current_hash != expected_current_sha256:
            raise FrontierError("stale frontier compare-and-swap hash")
        current = verify_frontier(
            policy=policy,
            outputs_root=outputs_root,
            repository_root=repository_root,
        )
        if current is not None and artifact["score"] <= current["score"]:
            raise FrontierError("candidate does not beat the verified current frontier")
        promotion_id = new_run_id("promote")
        history_path = frontier_root / "history" / f"{promotion_id}.json"
        run_staging = outputs_root / "runs/.staging" / promotion_id
        run_final = outputs_root / "runs" / promotion_id
        run_staging.mkdir(parents=True)
        score_ref = next(
            ArtifactRef.from_dict(item)
            for item in score_card["outputs"]
            if item["role"] == "score_artifact"
        )
        candidate = resolve_candidate(candidate_run_id, policy, outputs_root=outputs_root)
        verdict = {
            "schema_version": 1,
            "promotion_id": promotion_id,
            "status": "promoted",
            "candidate_run_id": candidate_run_id,
            "score_run_id": artifact["run_id"],
            "score": artifact["score"],
            "deltas": artifact["deltas"],
            "gates": artifact["gates"],
            "candidate_checkpoint": candidate.artifact,
            "score_artifact": score_ref.to_dict(),
            "policy_sha256": sha256_file(policy.path),
            "suite_hash": suite_hash(),
            "base_ref": base_ref,
            "integrity": integrity,
            "previous_frontier_sha256": actual_current_hash,
        }
        temporary_history = frontier_root / "history/.staging" / f"{promotion_id}.json"
        atomic_write_json(temporary_history, verdict)
        verdict_ref = _published_ref(
            temporary_history,
            history_path,
            repository_root,
        )
        observed_head = EventLog(outputs_root).validate()
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        card = {
            "schema_version": 1,
            "run_id": promotion_id,
            "run_type": "promote",
            "mode": "real",
            "status": "promoted",
            "started_at": now,
            "finished_at": now,
            "duration_seconds": time.monotonic() - started,
            "command": [
                "python",
                "-m",
                "boldt_posttrain.cli",
                "promote",
                "--candidate",
                candidate_run_id,
                "--base-ref",
                base_ref,
            ],
            "git": provenance.collect_git(base_ref, root=repository_root),
            "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
            "experiment": {
                "path": "none",
                "sha256": sha256_bytes(b"null"),
                "resolved_sha256": sha256_bytes(b"null"),
            },
            "inputs": [score_ref.to_dict(), candidate.artifact],
            "outputs": [verdict_ref.to_dict()],
            "model": candidate.to_dict(),
            "data": {"suite_hash": suite_hash()},
            "parameters": promotion,
            "hardware": provenance.collect_hardware(),
            "environment": {
                **provenance.collect_environment(),
                "event_head": {
                    key: observed_head[key] for key in ("sequence", "last_event_hash", "log_sha256")
                },
            },
            "parents": [artifact["run_id"], candidate_run_id],
            "compatibility_fingerprint": sha256_bytes(
                canonical_json_bytes({"policy": sha256_file(policy.path), "suite": suite_hash()})
            ),
            "error": None,
        }
        validate_run_card(card)
        atomic_write_json(run_staging / "run_card.json", card)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists() or run_final.exists():
            raise FrontierError("immutable promotion identifier already exists")
        os.replace(temporary_history, history_path)
        os.replace(run_staging, run_final)
        run_card_hash = sha256_file(run_final / "run_card.json")
        event_head = EventLog(outputs_root).append(
            "candidate_promoted",
            promotion_id,
            {
                "status": "promoted",
                "candidate_run_id": candidate_run_id,
                "score_run_id": artifact["run_id"],
                "run_card_sha256": run_card_hash,
            },
        )
        current_document = {
            "schema_version": 1,
            "promotion_id": promotion_id,
            "candidate_run_id": candidate_run_id,
            "score_run_id": artifact["run_id"],
            "score": artifact["score"],
            "policy_sha256": sha256_file(policy.path),
            "suite_hash": suite_hash(),
            "previous_frontier_sha256": actual_current_hash,
            "history_sha256": sha256_file(history_path),
            "run_card_sha256": run_card_hash,
            "event_head": {
                key: event_head[key] for key in ("sequence", "last_event_hash", "log_sha256")
            },
        }
        atomic_write_json(frontier_root / "current.json", current_document)
        return {
            "status": "promoted",
            "promotion_id": promotion_id,
            "candidate_run_id": candidate_run_id,
            "score": artifact["score"],
        }


def frontier_status(
    *, policy: Policy, outputs_root: Path = OUTPUTS, repository_root: Path = ROOT
) -> dict[str, Any]:
    current = verify_frontier(
        policy=policy,
        outputs_root=outputs_root,
        repository_root=repository_root,
    )
    if current is None:
        return {"status": "succeeded", "frontier": None, "verified_promotions": 0}
    verified = 0
    for history in sorted((outputs_root / "frontier/history").glob("*.json")):
        try:
            document = _load_json(history)
            promotion_id = history.stem
            card_path = outputs_root / "runs" / promotion_id / "run_card.json"
            card = _load_json(card_path)
            validate_run_card(card)
            verdict_refs = [
                ArtifactRef.from_dict(item)
                for item in card["outputs"]
                if item.get("role") == "promotion_verdict"
            ]
            if (
                card["run_type"] != "promote"
                or card["status"] != "promoted"
                or len(verdict_refs) != 1
                or verify_artifact_ref(verdict_refs[0], root=repository_root) != history.resolve()
            ):
                continue
            _event_anchor(outputs_root, promotion_id, sha256_file(card_path))
            artifact, _ = load_candidate_score(
                document["candidate_run_id"],
                policy,
                outputs_root=outputs_root,
                repository_root=repository_root,
            )
        except (ArtifactError, KeyError, ScoringError, FrontierError, OSError):
            continue
        if artifact["run_id"] == document.get("score_run_id") and artifact["status"] == "passed":
            verified += 1
    return {"status": "succeeded", "frontier": current, "verified_promotions": verified}


def run_cli(args) -> tuple[dict[str, Any], int]:
    policy = load_policy()
    if args.command == "promote":
        expected = current_frontier_hash(OUTPUTS / "frontier")
        result = promote_candidate(
            args.candidate,
            base_ref=args.base_ref,
            expected_current_sha256=expected,
            policy=policy,
        )
        return result, 0
    return frontier_status(policy=policy), 0
