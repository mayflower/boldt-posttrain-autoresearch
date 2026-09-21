#!/usr/bin/env bash
set -euo pipefail

# uv owns the environment. There is no Conda environment to protect and no
# hand-managed .venv to activate: uv.lock is the single source of truth.
#
# --locked never rewrites uv.lock. If pyproject.toml and the lock disagree, this
# aborts instead of quietly re-resolving -- the same fail-closed rule the rest of
# the repository applies to policy, data, and artifacts.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync --locked --all-extras

# uv run puts .venv/bin on PATH, which the merge and evaluation levers need to
# resolve mergekit-yaml and lm-eval via shutil.which.
uv run --locked python - <<'PY'
import importlib.metadata
import shutil
import sys

import boldt_posttrain  # noqa: F401  -- the project itself must be importable
import torch

expected_torch = importlib.metadata.version("torch")
if torch.__version__.split("+")[0] != expected_torch:
    sys.exit(f"locked torch {expected_torch}, imported {torch.__version__}")
if not torch.cuda.is_available():
    sys.exit("real levers require CUDA and never fall back to CPU")
for executable in ("mergekit-yaml", "lm-eval"):
    if shutil.which(executable) is None:
        sys.exit(f"{executable} is not on PATH")
print(f"ok: torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
PY
