#!/usr/bin/env bash
# Test all detectable backends by rebuilding and running the full test suite
# for each profile that is available in the current environment.
#
# Usage (from pyheom/ directory):
#   scripts/test_all_backends.sh [pytest-args...]
#
# Detection logic:
#   eigen  -- always tested
#   mkl    -- tested if ${MKLROOT}/include/mkl.h exists
#   cuda   -- tested if nvcc is in PATH after loading cuda/11.7
#   all    -- tested if both mkl and cuda are detected
#
# The original .so is restored after all profiles are tested.
# A summary table is printed at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/../.venv/bin/python"
SO_PATH="$ROOT_DIR/pyheom/$(ls "$ROOT_DIR/pyheom/pylibheom"*.so 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo 'pylibheom.so')"

MKL_HEADER="${MKLROOT}/include/mkl.h"

# ---------------------------------------------------------------------------
# Module init
# ---------------------------------------------------------------------------
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ---------------------------------------------------------------------------
# Detect available backends
# ---------------------------------------------------------------------------
HAS_MKL=false
HAS_CUDA=false

if [ -f "$MKL_HEADER" ]; then
    HAS_MKL=true
fi

module load cuda 2>/dev/null || true
if command -v nvcc &>/dev/null; then
    HAS_CUDA=true
fi

echo "=========================================="
echo " Backend detection"
echo "=========================================="
echo "  MKL  : $HAS_MKL"
echo "  CUDA : $HAS_CUDA"
echo ""

# Build profile list
PROFILES=("eigen")
$HAS_MKL  && PROFILES+=("mkl")
$HAS_CUDA && PROFILES+=("cuda")
($HAS_MKL && $HAS_CUDA) && PROFILES+=("all")

# ---------------------------------------------------------------------------
# Save original .so
# ---------------------------------------------------------------------------
SO_GLOB=("$ROOT_DIR"/pyheom/pylibheom*.so)
ORIGINAL_SO="${SO_GLOB[0]}"
SO_BACKUP="${ORIGINAL_SO}.bak_test_all"
cp "$ORIGINAL_SO" "$SO_BACKUP"
echo "Saved original .so -> $(basename "$SO_BACKUP")"
echo ""

# ---------------------------------------------------------------------------
# Run each profile
# ---------------------------------------------------------------------------
declare -A RESULTS

for PROFILE in "${PROFILES[@]}"; do
    echo "=========================================="
    echo " Profile: $PROFILE"
    echo "=========================================="

    # Rebuild
    if (cd "$ROOT_DIR" && bash scripts/build_backend.sh "$PROFILE" 2>&1); then
        BUILD_OK=true
    else
        BUILD_OK=false
        RESULTS[$PROFILE]="BUILD FAILED"
        echo "!!! Build failed for profile: $PROFILE"
        continue
    fi

    # Run tests
    cd "$ROOT_DIR"
    if "$VENV_PYTHON" -m pytest tests/ "$@" 2>&1; then
        RESULTS[$PROFILE]="PASSED"
    else
        RESULTS[$PROFILE]="TEST FAILED"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Restore original .so
# ---------------------------------------------------------------------------
cp "$SO_BACKUP" "$ORIGINAL_SO"
rm -f "$SO_BACKUP"
echo "Restored original .so"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo " Summary"
echo "=========================================="
for PROFILE in "${PROFILES[@]}"; do
    printf "  %-12s : %s\n" "$PROFILE" "${RESULTS[$PROFILE]:-NOT RUN}"
done
echo "=========================================="

# Exit non-zero if any profile failed
for PROFILE in "${PROFILES[@]}"; do
    if [[ "${RESULTS[$PROFILE]:-}" != "PASSED" ]]; then
        exit 1
    fi
done
