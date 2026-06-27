"""Console entrypoints (source-checkout convenience wrappers).

These power the ``pt-status`` / ``pt-report`` / ``pt-integrity`` console scripts declared in
pyproject. The canonical entrypoints remain ``python scripts/pt_*.py`` — these just import and call
the same script ``main()`` so an installed checkout has short commands too.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]


def _load(script_stem: str):
    path = ROOT / "scripts" / f"{script_stem}.py"
    spec = importlib.util.spec_from_file_location(script_stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main_status(argv: Optional[Sequence[str]] = None) -> int:
    return _load("pt_status").main(list(argv) if argv is not None else None)


def main_report(argv: Optional[Sequence[str]] = None) -> int:
    return _load("pt_report").main(list(argv) if argv is not None else None)


def main_integrity(argv: Optional[Sequence[str]] = None) -> int:
    return _load("check_posttrain_integrity").main(list(argv) if argv is not None else None)
