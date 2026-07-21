#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "boldtembed" || -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate the existing boldtembed Conda environment first" >&2
  exit 3
fi

if [[ "$(python -c 'import sys; print(sys.prefix)')" != "$CONDA_PREFIX" ]]; then
  echo "python does not belong to the active Conda environment" >&2
  exit 3
fi

before="$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
if [[ "$before" != "2.6.0+cu124 12.4" ]]; then
  echo "boldtembed must provide torch 2.6.0+cu124 with CUDA 12.4; found $before" >&2
  exit 3
fi

export VIRTUAL_ENV="$CONDA_PREFIX"
uv sync --active --all-extras --inexact --no-install-package torch --locked

after="$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
if [[ "$after" != "$before" ]]; then
  echo "protected Conda Torch changed during synchronization" >&2
  exit 5
fi

python -c 'import boldt_posttrain, torch; assert torch.cuda.is_available()'
