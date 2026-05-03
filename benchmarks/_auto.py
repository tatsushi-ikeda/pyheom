#  PyHEOM auto mode: memory estimation, warmup, and parameter tuning for benchmarks.

import os
import time
import numpy as np
from multiprocessing import cpu_count

from ._core import (
    build_solver, run_trial, available_engines,
    ALL_SPACES, ALL_FORMATS, SOLVER_KWARGS,
    T_FINAL, DT_CALLBACK, DT,
)
from pyheom._auto import _rss_bytes, _gpu_free_bytes, _thread_candidates


# ---------------------------------------------------------------------------
# Memory probing
# ---------------------------------------------------------------------------

def measure_rss_delta(fn, *args, **kwargs):
    """Call fn(*args) and return the RSS growth in bytes."""
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
# Auto selection (benchmark system)
# ---------------------------------------------------------------------------

def auto_select(engines=None, spaces=None, formats=None, solvers=None,
                n_trials=3, tune=True, verbose=True):
    """Discover engines, estimate memory, warmup, optionally tune threads.

    Returns a list of result dicts sorted by elapsed time.
    Each dict has keys: engine, space, format, solver, n_outer_threads,
    elapsed, rss_delta_mb.
    The recommended configuration is marked with key 'recommended': True.
    """
    gpu_free  = _gpu_free_bytes()
    engines   = engines if engines is not None else available_engines()
    spaces    = spaces  if spaces  is not None else ALL_SPACES
    formats   = formats if formats is not None else ALL_FORMATS
    solvers   = solvers if solvers is not None else list(SOLVER_KWARGS)

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
                    qme = build_solver(engine, space, fmt, solver)
                    if qme is None:
                        continue

                    rss_delta = measure_rss_delta(warmup, qme, solver)

                    if engine == 'cuda' and gpu_free is not None:
                        if rss_delta * 5 > gpu_free:
                            if verbose:
                                print(f'  SKIP {engine}/{space}/{fmt}/{solver}: '
                                      f'estimated {rss_delta*5/1024**2:.0f} MiB '
                                      f'> {gpu_free/1024**2:.0f} MiB GPU free')
                            continue

                    best_n = 1
                    if tune and engine in ('eigen', 'mkl'):
                        best_n, _ = tune_threads(engine, space, fmt, solver)
                        qme = build_solver(engine, space, fmt, solver,
                                           n_outer_threads=best_n)
                        if qme is None:
                            continue
                        warmup(qme, solver=solver)

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
