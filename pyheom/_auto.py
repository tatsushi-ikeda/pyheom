#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for license.
# ------------------------------------------------------------------------
#  PyHEOM auto-selection utilities shared between QMESolver.auto() and benchmarks.

import os
import subprocess
import time
from math import comb
import numpy as np
from multiprocessing import cpu_count


def _rss_bytes():
    """Current process RSS in bytes (Linux /proc; best-effort)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        pass
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _gpu_free_bytes():
    """Free GPU memory from nvidia-smi in bytes; None if unavailable."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip().split('\n')[0]
        return int(out) * 1024 * 1024
    except Exception:
        return None


def _host_available_bytes():
    """Host memory available without paging, in bytes; None if unavailable.

    Prefers psutil (cross-platform: Linux MemAvailable, macOS vm_stat,
    Windows GlobalMemoryStatusEx). Falls back to /proc/meminfo on Linux.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _estimate_n_hierarchy(noises, truncation_depth):
    """Estimate the HEOM hierarchy node count without building the solver."""
    K = sum(c.gamma.shape[0] for c in noises)
    return comb(K + truncation_depth, truncation_depth)


_ADO_SPARSE_BYTES_MULT     = 200
_HILBERT_LIOUVILLE_BYTES_MULT = 8


def _estimate_combination_bytes(space, fmt, n_level, n_hierarchy, dtype_size=16):
    """Conservative byte estimate for the dominant storage of a trial.

    Used by HEOMSolver.auto() to skip combinations that would overrun
    available memory before constructing them. Overestimates intentionally
    so the prune never skips a combination that would have built.
    """
    n_level_2 = n_level * n_level
    if space == 'ado':
        if fmt == 'dense':
            n_level_ado = n_hierarchy * n_level_2
            return n_level_ado * n_level_ado * dtype_size
        return _ADO_SPARSE_BYTES_MULT * n_hierarchy * n_level_2 * dtype_size
    return _HILBERT_LIOUVILLE_BYTES_MULT * n_hierarchy * n_level_2 * dtype_size


def _thread_candidates():
    """Candidate n_outer_threads values for Eigen thread tuning (1-D sweep).

    Uses cpu_count() as the upper bound regardless of OMP_NUM_THREADS, because
    the C++ OMP num_threads() clause overrides OMP_NUM_THREADS at runtime.
    """
    max_t = cpu_count()
    seen, candidates = set(), []
    for n in [1, 2, 4, 8, max_t]:
        if 1 <= n <= max_t and n not in seen:
            seen.add(n)
            candidates.append(n)
    return candidates


def _thread_pair_candidates():
    """Candidate (n_outer, n_inner) pairs for 2-D thread tuning.

    The pairs independently cover two regimes:
    - OMP-dominant (n_inner=1): for hilbert/liouville where the outer OMP
      loop over n_hrchy nodes is the bottleneck.
    - Inner-dominant (n_outer=1): for ADO where a single large gemv uses
      Eigen/MKL internal threads with no outer OMP loop.
    The oversubscribed ceiling (max, max) is included as a sanity check;
    Eigen/MKL may throttle internally in that case.

    Uses cpu_count() as the upper bound regardless of OMP_NUM_THREADS, because
    the C++ OMP num_threads() clause overrides OMP_NUM_THREADS at runtime.
    """
    max_t = cpu_count()
    pairs = set()
    for n in [1, 2, 4, 8, max_t]:
        if 1 <= n <= max_t:
            pairs.add((n, 1))      # OMP-dominant
            pairs.add((1, n))      # inner-dominant
    pairs.add((max_t, max_t))      # oversubscribed ceiling
    return sorted(pairs)


def _set_blas_threads(n):
    """Set MKL BLAS internal thread count; no-op if libmkl_rt.so is unavailable."""
    try:
        import ctypes
        lib = ctypes.CDLL('libmkl_rt.so', mode=ctypes.RTLD_GLOBAL)
        lib.MKL_Set_Num_Threads(ctypes.c_int(n))
    except OSError:
        pass


def _get_blas_threads():
    """Return current MKL BLAS thread count, or None if libmkl_rt.so is unavailable."""
    try:
        import ctypes
        lib = ctypes.CDLL('libmkl_rt.so', mode=ctypes.RTLD_GLOBAL)
        lib.MKL_Get_Max_Threads.restype = ctypes.c_int
        return int(lib.MKL_Get_Max_Threads())
    except OSError:
        return None


def _solve_kwargs(solver, dt):
    """Return kwargs for qme.solve() appropriate for the given solver."""
    if solver == 'rkdp':
        return {'dt': dt, 'atol': 1e-8, 'rtol': 1e-6}
    return {'dt': dt}
