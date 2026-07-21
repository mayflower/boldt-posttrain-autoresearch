"""Fail-closed Git, environment, hardware, timing, and fingerprint provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .artifacts import canonical_json_bytes, redact_command, sha256_file

ROOT = Path(__file__).resolve().parents[2]
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "safetensors",
    "huggingface-hub",
    "lm-eval",
    "mergekit",
)


class ProvenanceError(RuntimeError):
    """Required provenance could not be established."""


def _git(arguments: Sequence[str], *, root: Path = ROOT) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProvenanceError(f"git {' '.join(arguments)} failed: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ProvenanceError(f"git {' '.join(arguments)} failed ({result.returncode}): {message}")
    return result.stdout


def resolve_base_ref(base_ref: str, *, root: Path = ROOT) -> str:
    if not base_ref or base_ref.startswith("-"):
        raise ProvenanceError("base_ref must be a non-option Git revision")
    return _git(["rev-parse", "--verify", f"{base_ref}^{{commit}}"], root=root).decode().strip()


def collect_git(base_ref: str, *, root: Path = ROOT) -> dict[str, Any]:
    base = resolve_base_ref(base_ref, root=root)
    head = _git(["rev-parse", "HEAD"], root=root).decode().strip()
    status = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], root=root)
    tracked_diff = _git(["diff", "--binary", "HEAD"], root=root)
    staged_diff = _git(["diff", "--binary", "--cached", "HEAD"], root=root)
    committed_diff = _git(["diff", "--binary", f"{base}..{head}"], root=root)
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"], root=root).split(b"\0")
    untracked_records: list[dict[str, Any]] = []
    for raw in sorted(item for item in untracked if item):
        relative = raw.decode("utf-8", errors="surrogateescape")
        path = root / relative
        untracked_records.append(
            {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    diff_material = {
        "base_ref": base,
        "head": head,
        "committed": hashlib.sha256(committed_diff).hexdigest(),
        "tracked": hashlib.sha256(tracked_diff).hexdigest(),
        "staged": hashlib.sha256(staged_diff).hexdigest(),
        "untracked": untracked_records,
    }
    return {
        "base_ref": base,
        "head": head,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(canonical_json_bytes(diff_material)).hexdigest(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_hardware() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "os": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cuda_runtime": None,
        "gpus": [],
    }
    try:
        import torch
    except ImportError:
        return hardware
    hardware["cuda_runtime"] = torch.version.cuda
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            hardware["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "vram_bytes": props.total_memory,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    return hardware


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "packages": package_versions(),
    }


def fingerprint_model_files(model_dir: str | Path) -> dict[str, str]:
    root = Path(model_dir)
    required = {
        "model_config_sha256": root / "config.json",
        "tokenizer_sha256": root / "tokenizer.json",
        "tokenizer_config_sha256": root / "tokenizer_config.json",
        "chat_template_sha256": root / "chat_template.jinja",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ProvenanceError(f"model fingerprint inputs missing: {missing}")
    return {name: sha256_file(path) for name, path in required.items()}


def sanitize_command(command: Sequence[str]) -> list[str]:
    return redact_command(command)


@dataclass
class RunTimer:
    started_at: float
    started_monotonic: float

    @classmethod
    def start(cls) -> "RunTimer":
        return cls(time.time(), time.monotonic())

    def finish(self) -> dict[str, Any]:
        finished_at = time.time()
        from datetime import datetime, timezone

        return {
            "started_at": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            "finished_at": datetime.fromtimestamp(finished_at, timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - self.started_monotonic,
        }
