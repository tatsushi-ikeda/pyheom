#!/bin/bash
# Build distributable wheels using cibuildwheel + podman.
#
# Usage (from pyheom/ project root):
#   scripts/build_wheels.sh          # Eigen wheel only
#   scripts/build_wheels.sh --cuda   # CUDA wheel only
#   scripts/build_wheels.sh --all    # both
#
# Output: dist/wheelhouse/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CIBW="${VENV_DIR:-${PROJECT_DIR}/../.venv}/bin/cibuildwheel"
OUT="${PROJECT_DIR}/dist/wheelhouse"

BUILD_EIGEN=false
BUILD_CUDA=false

case "${1:-}" in
    --cuda) BUILD_CUDA=true ;;
    --all)  BUILD_EIGEN=true; BUILD_CUDA=true ;;
    "")     BUILD_EIGEN=true ;;
    *) echo "Usage: $0 [--cuda|--all]" >&2; exit 1 ;;
esac

mkdir -p "${OUT}"

if ${BUILD_EIGEN}; then
    echo "=== Building Eigen wheels ==="
    "${CIBW}" \
        --platform linux \
        --config-file "${PROJECT_DIR}/pyproject.toml" \
        --output-dir "${OUT}" \
        "${PROJECT_DIR}"
    echo "=== Eigen wheels written to ${OUT} ==="
fi

if ${BUILD_CUDA}; then
    if [ ! -d /opt/cuda/11.x ]; then
        echo "ERROR: /opt/cuda/11.x not found. Run: module load cuda/11.7" >&2
        exit 1
    fi
    echo "=== Building CUDA 11.7 wheels ==="
    "${CIBW}" \
        --platform linux \
        --config-file "${PROJECT_DIR}/pyproject.cuda.toml" \
        --output-dir "${OUT}" \
        "${PROJECT_DIR}"
    echo "=== CUDA wheels written to ${OUT} ==="
fi
