#!/usr/bin/env bash
# Run the full pyheom test suite, optionally rebuilding first.
#
# Usage (from pyheom/ directory):
#   scripts/run_tests.sh [PROFILE] [pytest-args...]
#
# If PROFILE is one of {eigen, mkl, cuda, all}, the extension is rebuilt
# for that profile before tests are run.  Any remaining arguments are passed
# directly to pytest.
#
# Machine-specific settings (VENV_LIBDIR, etc.) are read from
# scripts/local.env if it exists.  See scripts/local.env.template.
#
# Examples:
#   scripts/run_tests.sh                    # test current build, no rebuild
#   scripts/run_tests.sh cuda               # rebuild for CUDA, then test
#   scripts/run_tests.sh mkl -v -k K3      # rebuild for MKL, run only K3 tests

set -euo pipefail

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
# Detect if first argument is a backend profile
# ---------------------------------------------------------------------------
PROFILE=""
if [[ $# -gt 0 ]] && [[ " eigen mkl cuda all " == *" $1 "* ]]; then
    PROFILE="$1"
    shift
fi

# ---------------------------------------------------------------------------
# Rebuild if a profile was given
# ---------------------------------------------------------------------------
if [ -n "$PROFILE" ]; then
    echo "=========================================="
    echo " Rebuilding: profile = $PROFILE"
    echo "=========================================="
    (cd "$ROOT_DIR" && bash scripts/build_backend.sh "$PROFILE")
    echo ""
fi

# ---------------------------------------------------------------------------
# Report which backends are active
# ---------------------------------------------------------------------------
echo "=========================================="
echo " Backend status"
echo "=========================================="
"$VENV_PYTHON" - <<'EOF'
import pyheom.pylibheom as l
for name in ('eigen', 'mkl', 'cuda'):
    fn = getattr(l, f'{name}_is_supported')
    print(f"  {name:6s}: {fn()}")
EOF
echo ""

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo "=========================================="
echo " C++ tests (ctest)"
echo "=========================================="
_py_ver=$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
_platform=$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_platform())')
BUILD_TEMP="$ROOT_DIR/build/temp.${_platform}-${_py_ver}"
if [ -f "$BUILD_TEMP/CTestTestfile.cmake" ]; then
    # Prepend VENV_LIBDIR to LD_LIBRARY_PATH if set (needed when GTest's
    # libstdc++ is newer than the system compiler's).
    LD_LIBRARY_PATH="${VENV_LIBDIR:+${VENV_LIBDIR}:}${LD_LIBRARY_PATH:-}" \
        ctest --test-dir "$BUILD_TEMP" --output-on-failure
else
    echo "  (no CTestTestfile.cmake found -- C++ tests not built or not enabled)"
fi
echo ""

echo "=========================================="
echo " Python tests (pytest)"
echo "=========================================="
cd "$ROOT_DIR"
"$VENV_PYTHON" -m pytest tests/ "$@"
