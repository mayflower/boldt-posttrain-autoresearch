#!/usr/bin/env python3
"""Default-deny Git integrity check for the post-training trust boundary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_posttrain.policy import PolicyError, load_policy  # noqa: E402


class IntegrityError(RuntimeError):
    """Git state or policy could not be verified."""


def _glob_to_re(glob: str) -> re.Pattern[str]:
    output: list[str] = []
    index = 0
    while index < len(glob):
        if glob.startswith("**/", index):
            output.append("(?:.*/)?")
            index += 3
        elif glob.startswith("**", index):
            output.append(".*")
            index += 2
        elif glob[index] == "*":
            output.append("[^/]*")
            index += 1
        elif glob[index] == "?":
            output.append("[^/]")
            index += 1
        else:
            output.append(re.escape(glob[index]))
            index += 1
    return re.compile("^" + "".join(output) + "$")


def _normalize(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_globs(policy_path: str | Path | None = None) -> dict[str, list[str]]:
    policy = load_policy(policy_path or ROOT / "configs/posttrain/policy.json")
    return {
        "editable": list(policy.integrity["editable_globs"]),
        "protected": list(policy.integrity["protected_globs"]),
    }


def classify_paths(paths: Sequence[str], globs: dict[str, list[str]]) -> dict[str, list[str]]:
    editable_patterns = [_glob_to_re(item) for item in globs["editable"]]
    protected_patterns = [_glob_to_re(item) for item in globs["protected"]]
    result = {"editable": [], "protected": [], "other": []}
    for raw in paths:
        path = _normalize(raw)
        if not path:
            continue
        if any(pattern.fullmatch(path) for pattern in editable_patterns):
            result["editable"].append(path)
        elif any(pattern.fullmatch(path) for pattern in protected_patterns):
            result["protected"].append(path)
        else:
            result["other"].append(path)
    return {key: sorted(set(value)) for key, value in result.items()}


def evaluate(
    paths: Sequence[str],
    *,
    globs: dict[str, list[str]] | None = None,
    error: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    classifications = classify_paths(paths, globs or load_globs())
    violations = sorted(set(classifications["protected"] + classifications["other"]))
    if error:
        violations.append("<git-error>")
    return {
        "status": "fail" if violations else "pass",
        "violations": violations,
        "git_error": error,
        **classifications,
    }


def _git(arguments: Sequence[str], root: Path) -> bytes:
    try:
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrityError(f"git {' '.join(arguments)} failed: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise IntegrityError(f"git {' '.join(arguments)} failed ({result.returncode}): {stderr}")
    return result.stdout


def _nul_paths(output: bytes) -> list[str]:
    return [item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item]


def changed_paths(base_ref: str | None = None, *, root: Path = ROOT) -> list[str]:
    paths: list[str] = []
    if base_ref:
        if base_ref.startswith("-"):
            raise IntegrityError("base_ref must be a non-option Git revision")
        base = _git(["rev-parse", "--verify", f"{base_ref}^{{commit}}"], root).decode().strip()
        head = _git(["rev-parse", "HEAD"], root).decode().strip()
        paths.extend(_nul_paths(_git(["diff", "--name-only", "-z", f"{base}..{head}"], root)))
    paths.extend(_nul_paths(_git(["diff", "--name-only", "-z", "HEAD"], root)))
    paths.extend(_nul_paths(_git(["ls-files", "--others", "--exclude-standard", "-z"], root)))
    return sorted(set(paths))


def check(
    *, base_ref: str | None = None, root: Path = ROOT, policy_path: Path | None = None
) -> dict[str, Any]:
    try:
        globs = load_globs(policy_path)
        paths = changed_paths(base_ref, root=root)
        return evaluate(paths, globs=globs)
    except (IntegrityError, PolicyError) as exc:
        return evaluate([], globs={"editable": [], "protected": []}, error=str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="*", default=None)
    parser.add_argument("--base-ref")
    parser.add_argument("--policy", default=str(ROOT / "configs/posttrain/policy.json"))
    parser.add_argument("--format", choices=("json", "markdown", "text"), default="json")
    args = parser.parse_args(argv)
    if args.paths is None:
        result = check(base_ref=args.base_ref, policy_path=Path(args.policy))
    else:
        try:
            result = evaluate(args.paths, globs=load_globs(args.policy))
        except PolicyError as exc:
            result = evaluate([], globs={"editable": [], "protected": []}, error=str(exc))
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.format == "markdown":
        print(f"# Post-training integrity: **{result['status'].upper()}**")
        for category in ("editable", "protected", "other"):
            print(f"- {category}: {result[category] or 'none'}")
        if result["git_error"]:
            print(f"- git_error: {result['git_error']}")
    else:
        print(f"Post-training integrity: {result['status'].upper()}")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 5


if __name__ == "__main__":
    raise SystemExit(main())
