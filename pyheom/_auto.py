#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------
#  PyHEOM auto-selection utilities shared between QMESolver.auto() and benchmarks.

import os
import subprocess
import time
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


def _thread_candidates():
    """Candidate n_outer_threads values to sweep during thread tuning."""
    max_t = int(os.environ.get('OMP_NUM_THREADS', cpu_count()))
    seen, candidates = set(), []
    for n in [1, 2, 4, 8, max_t]:
        if 1 <= n <= max_t and n not in seen:
            seen.add(n)
            candidates.append(n)
    return candidates


def _solve_kwargs(solver, dt):
    """Return kwargs for qme.solve() appropriate for the given solver."""
    if solver == 'rkdp':
        return {'dt': dt, 'atol': 1e-8, 'rtol': 1e-6}
    return {'dt': dt}
