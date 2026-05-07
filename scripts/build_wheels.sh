#!/bin/bash
# Build distributable wheels using cibuildwheel + podman.
#
# Usage (from pyheom/ project root):
#   scripts/build_wheels.sh            # Eigen wheel only
#   scripts/build_wheels.sh --cuda     # CUDA 11.7 wheel only
#   scripts/build_wheels.sh --cuda12   # CUDA 12.x wheel only
#   scripts/build_wheels.sh --all      # Eigen + CUDA 11.7 + CUDA 12.x
#
# Output: dist/wheelhouse/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CIBW="${VENV_DIR:-${PROJECT_DIR}/../.venv}/bin/cibuildwheel"
OUT="${PROJECT_DIR}/dist/wheelhouse"

BUILD_EIGEN=false
BUILD_CUDA=false
BUILD_CUDA12=false

case "${1:-}" in
    --cuda)   BUILD_CUDA=true ;;
    --cuda12) BUILD_CUDA12=true ;;
    --all)    BUILD_EIGEN=true; BUILD_CUDA=true; BUILD_CUDA12=true ;;
    "")       BUILD_EIGEN=true ;;
    *) echo "Usage: $0 [--cuda|--cuda12|--all]" >&2; exit 1 ;;
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
    CUDA11_DIR="${CUDA_HOME:-}"
    if [ -z "${CUDA11_DIR}" ] || [ ! -d "${CUDA11_DIR}" ]; then
        CUDA11_DIR="$(ls -d /opt/cuda/11.* 2>/dev/null | sort -V | tail -1)"
    fi
    if [ -z "${CUDA11_DIR}" ] || [ ! -d "${CUDA11_DIR}" ]; then
        echo "ERROR: no CUDA 11 installation found." >&2
        echo "  Run: module load cuda/11.x  or set CUDA_HOME=/path/to/cuda11" >&2
        exit 1
    fi
    echo "=== Building CUDA 11.7 wheels using ${CUDA11_DIR} ==="
    CUDA11_TOML="${PROJECT_DIR}/pyproject.cuda.toml"
    TMP_CUDA11_TOML="$(mktemp /tmp/pyproject.cuda11.XXXXXX.toml)"
    sed "s|/opt/cuda/11.x|${CUDA11_DIR}|g" "${CUDA11_TOML}" > "${TMP_CUDA11_TOML}"
    "${CIBW}" \
        --platform linux \
        --config-file "${TMP_CUDA11_TOML}" \
        --output-dir "${OUT}" \
        "${PROJECT_DIR}"
    rm -f "${TMP_CUDA11_TOML}"
    echo "=== CUDA 11.7 wheels written to ${OUT} ==="
fi

if ${BUILD_CUDA12}; then
    # Find the CUDA 12 installation: prefer the path set by module load, else
    # fall back to the newest /opt/cuda/12.* directory.
    CUDA12_DIR="${CUDA_HOME:-}"
    if [ -z "${CUDA12_DIR}" ] || [ ! -d "${CUDA12_DIR}" ]; then
        CUDA12_DIR="$(ls -d /opt/cuda/12.* 2>/dev/null | sort -V | tail -1)"
    fi
    if [ -z "${CUDA12_DIR}" ] || [ ! -d "${CUDA12_DIR}" ]; then
        echo "ERROR: no CUDA 12 installation found." >&2
        echo "  Run: module load cuda/12.x  or set CUDA_HOME=/opt/cuda/12.x" >&2
        exit 1
    fi
    echo "=== Building CUDA 12.x wheels using ${CUDA12_DIR} ==="
    # Substitute the actual CUDA 12 path into the config before running cibuildwheel.
    CUDA12_TOML="${PROJECT_DIR}/pyproject.cuda12.toml"
    TMP_TOML="$(mktemp /tmp/pyproject.cuda12.XXXXXX.toml)"
    sed "s|/opt/cuda/12.x|${CUDA12_DIR}|g" "${CUDA12_TOML}" > "${TMP_TOML}"
    "${CIBW}" \
        --platform linux \
        --config-file "${TMP_TOML}" \
        --output-dir "${OUT}" \
        "${PROJECT_DIR}"
    rm -f "${TMP_TOML}"
    echo "=== CUDA 12.x wheels written to ${OUT} ==="
fi
