#!/usr/bin/env python3
"""Guard the post-training AutoResearch protected surfaces (pure stdlib).

PROTECTED SURFACE. The loop may edit ONLY the editable globs (experiment configs, current.json,
and notes under docs/experiments/). Everything that defines how a trial is JUDGED — scoring, gates,
eval scripts, leakage checks, committed baselines, and the governance docs — is protected. This
classifies the changed paths (from ``git status``, optionally also everything committed since a
``--base-ref``) and FAILS if any protected surface was touched.

The editable/protected globs are read from ``configs/posttrain/base.json`` (single source of truth,
itself protected), so this guard and the documented policy can never drift apart.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "posttrain" / "base.json"

# Fallback globs if base.json is unreadable (fail-closed: still protect the critical surfaces).
_FALLBACK_EDITABLE = ["configs/posttrain/current.json", "configs/posttrain/experiments/*.json",
                      "docs/experiments/*.md"]
_FALLBACK_PROTECTED = ["data/eval/**", "scripts/pt_eval.py", "scripts/pt_score.py",
                       "scripts/pt_promote.py", "scripts/check_posttrain_integrity.py",
                       "src/boldt_posttrain/scoring.py", "outputs/posttrain/baseline/**",
                       "CLAUDE.md", "AUTORESEARCH_POSTTRAIN.md"]


def load_globs() -> Dict[str, List[str]]:
    try:
        cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        integ = cfg.get("integrity", {})
        editable = integ.get("editable_globs") or _FALLBACK_EDITABLE
        protected = integ.get("protected_globs") or _FALLBACK_PROTECTED
    except Exception:
        editable, protected = _FALLBACK_EDITABLE, _FALLBACK_PROTECTED
    # The scorer module is protected even though it lives in src/ (the gate's real definition).
    protected = sorted(set(protected) | {"src/boldt_posttrain/scoring.py"})
    return {"editable": editable, "protected": protected}


def _glob_to_re(glob: str) -> re.Pattern:
    out: List[str] = []
    i, n = 0, len(glob)
    while i < n:
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _norm(path: str) -> str:
    return path.strip().lstrip("./").replace("\\", "/")


def classify_paths(paths: List[str], globs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    editable_re = [_glob_to_re(g) for g in globs["editable"]]
    protected_re = [_glob_to_re(g) for g in globs["protected"]]
    editable, protected, other = [], [], []
    for raw in paths:
        p = _norm(raw)
        if not p:
            continue
        if any(rx.match(p) for rx in editable_re):
            editable.append(p)
        elif any(rx.match(p) for rx in protected_re):
            protected.append(p)
        else:
            other.append(p)
    return {"editable": sorted(set(editable)), "protected": sorted(set(protected)),
            "other": sorted(set(other))}


def _porcelain_paths() -> List[str]:
    paths: List[str] = []
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=15)
    except Exception:
        return paths
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        body = line[3:]
        if " -> " in body:  # rename: "old -> new"
            old, new = body.split(" -> ", 1)
            paths.extend([old.strip(), new.strip()])
        else:
            paths.append(body.strip())
    return paths


def changed_paths(base_ref: Optional[str] = None) -> List[str]:
    paths = _porcelain_paths()
    if base_ref:
        try:
            out = subprocess.run(["git", "diff", "--name-only", base_ref], cwd=str(ROOT),
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                paths += [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        except Exception:
            pass
    return paths


def evaluate(paths: List[str], strict: bool = False,
             globs: Optional[Dict[str, List[str]]] = None) -> Dict[str, object]:
    cls = classify_paths(paths, globs or load_globs())
    violations = list(cls["protected"])
    if strict:
        violations += cls["other"]
    return {"status": "pass" if not violations else "fail",
            "violations": sorted(set(violations)), **cls}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="explicit changed paths to check (default: read from git)")
    ap.add_argument("--base-ref", default=None,
                    help="also vet everything committed since this ref (e.g. the loop's start "
                         "commit) so committing a protected edit cannot bypass the gate")
    ap.add_argument("--strict", action="store_true",
                    help="also fail if anything other than the editable surface changed")
    ap.add_argument("--format", choices=["json", "markdown", "text"], default="text")
    args = ap.parse_args(argv)

    paths = args.paths if args.paths is not None else changed_paths(args.base_ref)
    result = evaluate(paths, strict=args.strict)

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.format == "markdown":
        print(f"# Post-training integrity: **{result['status'].upper()}**\n")
        if result["editable"]:
            print("- editable touched: " + ", ".join(f"`{p}`" for p in result["editable"]))
        if result["violations"]:
            print("- **PROTECTED surfaces touched (not allowed):**")
            for v in result["violations"]:
                print(f"  - ✗ `{v}`")
        elif not paths:
            print("- (no changed paths detected)")
    else:
        print(f"Post-training integrity: {result['status'].upper()}")
        if result["editable"]:
            print("  editable touched: " + ", ".join(result["editable"]))
        if result["violations"]:
            print("  PROTECTED surfaces touched (not allowed):")
            for v in result["violations"]:
                print(f"    x {v}")
        elif not paths:
            print("  (no changed paths detected)")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
