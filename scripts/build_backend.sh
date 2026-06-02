#!/usr/bin/env bash
# Build pylibheom for a specific backend profile.
#
# Usage (from pyheom/ directory):
#   scripts/build_backend.sh PROFILE
#
# PROFILE values:
#   eigen           Eigen only (default)
#   mkl             Eigen + Intel MKL
#   cuda            Eigen + CUDA
#   all             Eigen + MKL + CUDA
#
# Machine-specific settings (MKL path, CUDA arch, etc.) are read from
# scripts/local.env if it exists.  See scripts/local.env.template.
#
# After a successful build the source-tree .so is updated in-place so that
# the editable install ("pip install -e .") picks it up immediately.

set -euo pipefail

PROFILE="${1:-eigen}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/../.venv/bin/python"

# ---------------------------------------------------------------------------
# Load machine-specific settings (gitignored)
# ---------------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/local.env" ]; then
    source "$SCRIPT_DIR/local.env"
fi

# ---------------------------------------------------------------------------
# Derive build directories from the active Python interpreter
# ---------------------------------------------------------------------------
_py_ver=$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
_platform=$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_platform())')
BUILD_LIB="$ROOT_DIR/build/lib.${_platform}-${_py_ver}/pyheom"
BUILD_TEMP="$ROOT_DIR/build/temp.${_platform}-${_py_ver}"
SO_DEST="$ROOT_DIR/pyheom"

CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-70}"

# ---------------------------------------------------------------------------
# Module initialization (non-interactive shells don't source .bashrc)
# ---------------------------------------------------------------------------
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ---------------------------------------------------------------------------
# Profile-specific settings
# ---------------------------------------------------------------------------
USE_MKL=OFF
USE_CUDA=OFF
EXTRA_CMAKE_ARGS=""

case "$PROFILE" in
    eigen)
        echo ">>> Backend profile: Eigen only"
        ;;
    mkl)
        echo ">>> Backend profile: Eigen + MKL"
        if [ -z "${MKLROOT:-}" ]; then
            echo "ERROR: MKLROOT is not set." >&2
            echo "  Run 'module load intel/...' or set MKLROOT in scripts/local.env" >&2
            exit 1
        fi
        MKL_INCLUDE="$MKLROOT/include"
        MKL_LIBDIR="$MKLROOT/lib/intel64"
        MKL_RT="$MKL_LIBDIR/libmkl_rt.so"
        if [ ! -f "$MKL_INCLUDE/mkl.h" ]; then
            echo "ERROR: MKL headers not found at $MKL_INCLUDE" >&2
            exit 1
        fi
        if [ ! -f "$MKL_RT" ]; then
            echo "ERROR: libmkl_rt.so not found at $MKL_RT" >&2
            exit 1
        fi
        export CPATH="$MKL_INCLUDE:${CPATH:-}"
        export LD_LIBRARY_PATH="$MKL_LIBDIR:${LD_LIBRARY_PATH:-}"
        USE_MKL=ON
        EXTRA_CMAKE_ARGS="-DBLAS_LIBRARIES=$MKL_RT"
        ;;
    cuda)
        echo ">>> Backend profile: Eigen + CUDA"
        module load cuda 2>/dev/null || true
        if ! command -v nvcc &>/dev/null; then
            echo "ERROR: nvcc not found; load the CUDA module first" >&2
            exit 1
        fi
        USE_CUDA=ON
        EXTRA_CMAKE_ARGS="-DCUDA_ARCH_LIST=$CUDA_ARCH_LIST"
        ;;
    all)
        echo ">>> Backend profile: Eigen + MKL + CUDA"
        if [ -z "${MKLROOT:-}" ]; then
            echo "ERROR: MKLROOT is not set." >&2
            echo "  Run 'module load intel/...' or set MKLROOT in scripts/local.env" >&2
            exit 1
        fi
        MKL_INCLUDE="$MKLROOT/include"
        MKL_LIBDIR="$MKLROOT/lib/intel64"
        MKL_RT="$MKL_LIBDIR/libmkl_rt.so"
        if [ ! -f "$MKL_INCLUDE/mkl.h" ]; then
            echo "ERROR: MKL headers not found at $MKL_INCLUDE" >&2
            exit 1
        fi
        if [ ! -f "$MKL_RT" ]; then
            echo "ERROR: libmkl_rt.so not found at $MKL_RT" >&2
            exit 1
        fi
        module load cuda 2>/dev/null || true
        if ! command -v nvcc &>/dev/null; then
            echo "ERROR: nvcc not found; load the CUDA module first" >&2
            exit 1
        fi
        export CPATH="$MKL_INCLUDE:${CPATH:-}"
        export LD_LIBRARY_PATH="$MKL_LIBDIR:${LD_LIBRARY_PATH:-}"
        USE_MKL=ON
        USE_CUDA=ON
        EXTRA_CMAKE_ARGS="-DBLAS_LIBRARIES=$MKL_RT -DCUDA_ARCH_LIST=$CUDA_ARCH_LIST"
        ;;
    *)
        echo "ERROR: Unknown profile '$PROFILE'" >&2
        echo "Valid profiles: eigen  mkl  cuda  all" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Clear CMake cache so the new settings take effect
# ---------------------------------------------------------------------------
echo ">>> Clearing CMake cache ..."
rm -f "$BUILD_TEMP/CMakeCache.txt"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo ">>> Building pylibheom (profile=$PROFILE) ..."
CMAKE_ARGS="-DLIBHEOM_ENABLE_EIGEN=ON -DLIBHEOM_ENABLE_MKL=$USE_MKL -DLIBHEOM_ENABLE_CUDA=$USE_CUDA $EXTRA_CMAKE_ARGS" \
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    "$VENV_PYTHON" setup.py build_ext 2>&1
# ^^^ build_ext without --inplace puts .so into build/lib.<platform>/pyheom/

# ---------------------------------------------------------------------------
# Install .so into source tree (editable install reads from here)
# ---------------------------------------------------------------------------
SO_FILE=$(ls "$BUILD_LIB"/pylibheom*.so 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: Built .so not found in $BUILD_LIB" >&2
    exit 1
fi
echo ">>> Copying $SO_FILE -> $SO_DEST/"
cp "$SO_FILE" "$SO_DEST/"

echo ">>> Build complete. Backend profile: $PROFILE"
echo "    eigen_is_supported : $("$VENV_PYTHON" -c 'import pyheom.pylibheom as l; print(l.eigen_is_supported())')"
echo "    mkl_is_supported   : $("$VENV_PYTHON" -c 'import pyheom.pylibheom as l; print(l.mkl_is_supported())')"
echo "    cuda_is_supported  : $("$VENV_PYTHON" -c 'import pyheom.pylibheom as l; print(l.cuda_is_supported())')"
