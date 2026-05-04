#  PyHEOM auto mode: memory estimation, warmup, and parameter tuning for benchmarks.

import os
import time
import numpy as np
from multiprocessing import cpu_count

from ._core import (
    build_solver, run_trial, available_engines,
    ALL_SPACES, ALL_FORMATS,
    T_FINAL, DT_CALLBACK, DT, _H,
)
from pyheom._auto import _rss_bytes, _gpu_free_bytes, _thread_candidates

_BENCH_N_LEVEL = _H().shape[0]


def _engine_unrollings(engine, user_unrollings):
    """Return the unrolling list to test for a given engine.

    When user_unrollings is None (auto), both True and False are tested for
    eigen when the benchmark system size supports static template specialisation;
    all other engines always use dynamic templates so only True is tested.
    """
    if user_unrollings is not None:
        return user_unrollings
    if engine == 'eigen' and _BENCH_N_LEVEL in [2, 3, 4]:
        return [True, False]
    return [True]


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


def warmup(qme):
    """Run a short trial to populate instruction caches. Returns elapsed seconds."""
    return run_trial(qme, t_final=_WARMUP_STEPS * _WARMUP_DT_CALLBACK,
                     dt_callback=_WARMUP_DT_CALLBACK)


# ---------------------------------------------------------------------------
# Thread tuning (CPU engines only)
# ---------------------------------------------------------------------------

def tune_threads(engine, space, fmt, solver, unrolling=True, n_trials=2):
    """Try several n_outer_threads values and return the best (n_threads, elapsed)."""
    best_n, best_t = 1, float('inf')
    for n in _thread_candidates():
        qme = build_solver(engine, space, fmt, solver, unrolling=unrolling,
                           n_outer_threads=n)
        if qme is None:
            continue
        warmup(qme)
        elapsed = min(run_trial(qme) for _ in range(n_trials))
        if elapsed < best_t:
            best_n, best_t = n, elapsed
    return best_n, best_t


# ---------------------------------------------------------------------------
# Auto selection (benchmark system)
# ---------------------------------------------------------------------------

def auto_select(engine=None, engines=None,
                space=None,  spaces=None,
                format=None, formats=None,
                unrollings=None, n_trials=3, tune=True, verbose=True):
    """Discover engines, estimate memory, warmup, optionally tune threads.

    Returns a list of result dicts sorted by elapsed time.
    Each dict has keys: engine, space, format, solver, unrolling, n_outer_threads,
    elapsed, rss_delta_mb.
    The recommended configuration is marked with key 'recommended': True.
    The ODE solver is fixed to 'lsrk4' and is not swept.
    """
    gpu_free = _gpu_free_bytes()

    # Singular-form arguments override the plural list forms.
    if engine is not None:
        engines = [engine]
    if space is not None:
        spaces = [space]
    if format is not None:
        formats = [format]

    engines = engines if engines is not None else available_engines()
    spaces  = spaces  if spaces  is not None else ALL_SPACES
    formats = formats if formats is not None else ALL_FORMATS
    solver  = 'lsrk4'

    if verbose:
        print(f'Available engines : {engines}')
        if gpu_free is not None:
            print(f'GPU free memory   : {gpu_free / 1024**3:.1f} GiB')
        print()

    results = []

    for eng in engines:
        engine_unrollings = _engine_unrollings(eng, unrollings)
        for sp in spaces:
            for fmt in formats:
                for unrolling in engine_unrollings:
                    qme = build_solver(eng, sp, fmt, solver,
                                       unrolling=unrolling)
                    if qme is None:
                        continue

                    rss_delta = measure_rss_delta(warmup, qme)

                    if eng == 'cuda' and gpu_free is not None:
                        if rss_delta * 5 > gpu_free:
                            if verbose:
                                unrl_tag = 'on' if unrolling else 'off'
                                print(f'  SKIP {eng}/{sp}/{fmt}'
                                      f'/unroll={unrl_tag}: '
                                      f'estimated {rss_delta*5/1024**2:.0f} MiB '
                                      f'> {gpu_free/1024**2:.0f} MiB GPU free')
                            continue

                    best_n = 1
                    if tune and eng in ('eigen', 'mkl'):
                        best_n, _ = tune_threads(eng, sp, fmt, solver,
                                                 unrolling=unrolling)
                        qme = build_solver(eng, sp, fmt, solver,
                                           unrolling=unrolling,
                                           n_outer_threads=best_n)
                        if qme is None:
                            continue
                        warmup(qme)

                    times = [run_trial(qme) for _ in range(n_trials)]
                    elapsed = float(np.median(times))

                    entry = dict(
                        engine=eng, space=sp, format=fmt, solver=solver,
                        unrolling=unrolling, n_outer_threads=best_n,
                        elapsed=elapsed, rss_delta_mb=rss_delta / 1024**2,
                    )
                    results.append(entry)

                    if verbose:
                        unrl_tag = 'on' if unrolling else 'off'
                        print(f'  {eng:6s} {sp:10s} {fmt:7s} '
                              f'unroll={unrl_tag} threads={best_n:<3d} '
                              f'{elapsed:.3f}s  '
                              f'mem~{rss_delta/1024**2:.1f}MiB')

    results.sort(key=lambda x: x['elapsed'])
    if results:
        results[0]['recommended'] = True
    return results
