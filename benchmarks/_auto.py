#  PyHEOM auto mode: memory estimation, warmup, and parameter tuning.

import os
import subprocess
import time
import numpy as np
from multiprocessing import cpu_count

from ._core import (
    build_solver, run_trial, available_engines,
    ALL_SPACES, ALL_FORMATS, ALL_SOLVERS,
    T_FINAL, DT_CALLBACK, DT,
)


# ---------------------------------------------------------------------------
# Memory probing
# ---------------------------------------------------------------------------

def _rss_bytes():
    """Current process resident set size in bytes (Linux /proc; best-effort)."""
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
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip().split('\n')[0]
        return int(out) * 1024 * 1024
    except Exception:
        return None


def measure_rss_delta(fn, *args, **kwargs):
    """Call fn(*args) and return the RSS growth in bytes.

    Not exact (GC and OS effects) but sufficient for order-of-magnitude
    memory guards.
    """
    before = _rss_bytes()
    fn(*args, **kwargs)
    return max(0, _rss_bytes() - before)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

_WARMUP_STEPS = 5
_WARMUP_DT_CALLBACK = DT_CALLBACK


def warmup(qme, solver='lsrk4'):
    """Run a short trial to populate instruction caches. Returns elapsed seconds."""
    return run_trial(qme, solver=solver,
                     t_final=_WARMUP_STEPS * _WARMUP_DT_CALLBACK,
                     dt_callback=_WARMUP_DT_CALLBACK)


# ---------------------------------------------------------------------------
# Thread tuning (CPU engines only)
# ---------------------------------------------------------------------------

def _thread_candidates():
    max_t = int(os.environ.get('OMP_NUM_THREADS', cpu_count()))
    seen, candidates = set(), []
    for n in [1, 2, 4, 8, max_t]:
        if 1 <= n <= max_t and n not in seen:
            seen.add(n)
            candidates.append(n)
    return candidates


def tune_threads(engine, space, fmt, solver, n_trials=2):
    """Try several n_outer_threads values and return the best (n_threads, elapsed)."""
    best_n, best_t = 1, float('inf')
    for n in _thread_candidates():
        qme = build_solver(engine, space, fmt, solver, n_outer_threads=n)
        if qme is None:
            continue
        warmup(qme, solver=solver)
        elapsed = min(run_trial(qme, solver=solver) for _ in range(n_trials))
        if elapsed < best_t:
            best_n, best_t = n, elapsed
    return best_n, best_t


# ---------------------------------------------------------------------------
# Auto selection
# ---------------------------------------------------------------------------

def auto_select(engines=None, spaces=None, formats=None, solvers=None,
                n_trials=3, tune=True, verbose=True):
    """Discover engines, estimate memory, warmup, optionally tune threads.

    Returns a list of result dicts sorted by elapsed time.
    Each dict has keys: engine, space, format, solver, n_outer_threads,
    elapsed, rss_delta_mb.
    The recommended configuration is marked with key 'recommended': True.
    """
    gpu_free = _gpu_free_bytes()
    engines  = engines if engines is not None else available_engines()
    spaces   = spaces  if spaces  is not None else ALL_SPACES
    formats  = formats if formats is not None else ALL_FORMATS
    solvers  = solvers if solvers is not None else ALL_SOLVERS

    if verbose:
        print(f'Available engines : {engines}')
        if gpu_free is not None:
            print(f'GPU free memory   : {gpu_free / 1024**3:.1f} GiB')
        print()

    results = []

    for engine in engines:
        for space in spaces:
            for fmt in formats:
                for solver in solvers:
                    # 1. Try construction (skip if not compiled)
                    qme = build_solver(engine, space, fmt, solver)
                    if qme is None:
                        continue

                    # 2. Measure RSS growth from warmup as memory estimate
                    rss_delta = measure_rss_delta(warmup, qme, solver)

                    # 3. GPU memory guard (conservative: 5x for C++ device buffers)
                    if engine == 'cuda' and gpu_free is not None:
                        if rss_delta * 5 > gpu_free:
                            if verbose:
                                print(f'  SKIP {engine}/{space}/{fmt}/{solver}: '
                                      f'estimated {rss_delta*5/1024**2:.0f} MiB '
                                      f'> {gpu_free/1024**2:.0f} MiB GPU free')
                            continue

                    # 4. Thread tuning for CPU engines
                    best_n = 1
                    if tune and engine in ('eigen', 'mkl'):
                        best_n, _ = tune_threads(engine, space, fmt, solver)
                        qme = build_solver(engine, space, fmt, solver,
                                           n_outer_threads=best_n)
                        if qme is None:
                            continue
                        warmup(qme, solver=solver)

                    # 5. Timing trials
                    times = [run_trial(qme, solver=solver) for _ in range(n_trials)]
                    elapsed = float(np.median(times))

                    entry = dict(
                        engine=engine, space=space, format=fmt, solver=solver,
                        n_outer_threads=best_n, elapsed=elapsed,
                        rss_delta_mb=rss_delta / 1024**2,
                    )
                    results.append(entry)

                    if verbose:
                        print(f'  {engine:6s} {space:10s} {fmt:7s} {solver:6s} '
                              f'threads={best_n:<3d} '
                              f'{elapsed:.3f}s  '
                              f'mem~{rss_delta/1024**2:.1f}MiB')

    results.sort(key=lambda x: x['elapsed'])
    if results:
        results[0]['recommended'] = True
    return results
