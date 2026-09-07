#!/usr/bin/env bash
set -euo pipefail

HABITUS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HABITUS_ROCM_VENV="${HABITUS_PROJECT_ROOT}/.venv-rocm"
HABITUS_PYTHON_BIN="${HABITUS_PYTHON_BIN:-python3}"

"${HABITUS_PYTHON_BIN}" -m venv "${HABITUS_ROCM_VENV}"
"${HABITUS_ROCM_VENV}/bin/python" -m pip install --upgrade pip
"${HABITUS_ROCM_VENV}/bin/python" -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'torch[device-gfx1103]==2.12.0+rocm7.14.1'
"${HABITUS_ROCM_VENV}/bin/python" -m pip install 'numpy>=2' 'pytest>=8'

echo "ROCm cortex environment installed at ${HABITUS_ROCM_VENV}"
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "Current login lacks the render group; GPU access will remain unavailable."
  echo "See docs/DEVELOPMENTAL_CORTEX.md before changing host permissions."
fi

if [[ "${1:-}" == "--probe" ]]; then
  cd "${HABITUS_PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
    "${HABITUS_ROCM_VENV}/bin/python" \
    experiments/graph_native_live/rocm_cortex_probe.py
fi
