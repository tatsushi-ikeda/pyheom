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
from pyheom._auto import (_rss_bytes, _gpu_free_bytes,
                          _thread_candidates, _thread_pair_candidates)

_BENCH_N_LEVEL = _H().shape[0]


def _engine_unrollings(engine, user_unrollings):
    """Return the unrolling list to test for a given engine.

    When user_unrollings is None (auto), both True and False are tested for
    eigen when the benchmark system size supports static template specialisation;
    all other engines always use dynamic templates so only True is tested.
    """
    if user_unrollings is not None:
        return user_unrollings
    if engine == 'Eigen' and _BENCH_N_LEVEL in [2, 3, 4]:
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

_N_TUNE_STEPS = 10  # ODE steps per tuning trial; sufficient for relative ranking


def _tune_trial(qme, n_steps=_N_TUNE_STEPS):
    """Run n_steps lsrk4 steps; dt_callback=DT so each step is one output."""
    return run_trial(qme, t_final=n_steps * DT, dt_callback=DT)


def tune_threads(engine, space, fmt, solver='lsrk4', unrolling=True,
                 n_trials=2, n_tune_steps=_N_TUNE_STEPS, verbose=False):
    """Find the best (n_outer, n_inner) thread pair and return (n_outer, n_inner, elapsed).

    Uses short n_tune_steps-step runs (lsrk4) for ranking so tuning is fast.
    The returned elapsed time is from the short run and not comparable to T_FINAL timings.
    """
    best_outer, best_inner, best_t = 1, 1, float('inf')
    for n_outer, n_inner in _thread_pair_candidates():
        qme = build_solver(engine, space, fmt, 'lsrk4', unrolling=unrolling,
                           n_outer_threads=n_outer, n_inner_threads=n_inner)
        if qme is None:
            continue
        warmup(qme)
        elapsed = min(_tune_trial(qme, n_tune_steps) for _ in range(n_trials))
        if verbose:
            mark = ''
            if elapsed < best_t:
                mark = ' <'
            print(f'    tune omp={n_outer} inner={n_inner}: {elapsed*1e3:.1f}ms{mark}',
                  flush=True)
        if elapsed < best_t:
            best_outer, best_inner, best_t = n_outer, n_inner, elapsed
    return best_outer, best_inner, best_t


# ---------------------------------------------------------------------------
# Auto selection (benchmark system)
# ---------------------------------------------------------------------------

def auto_select(engine=None, engines=None,
                space=None,  spaces=None,
                format=None, formats=None,
                unrollings=None, n_trials=3, tune=True,
                n_tune_steps=_N_TUNE_STEPS, verbose=True):
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

                    best_outer, best_inner = 1, 1
                    if tune and eng in ('Eigen', 'MKL'):
                        best_outer, best_inner, _ = tune_threads(
                            eng, sp, fmt, solver, unrolling=unrolling,
                            n_tune_steps=n_tune_steps)
                        qme = build_solver(eng, sp, fmt, solver,
                                           unrolling=unrolling,
                                           n_outer_threads=best_outer,
                                           n_inner_threads=best_inner)
                        if qme is None:
                            continue
                        warmup(qme)

                    times = [run_trial(qme) for _ in range(n_trials)]
                    elapsed = float(np.median(times))

                    entry = dict(
                        engine=eng, space=sp, format=fmt, solver=solver,
                        unrolling=unrolling,
                        n_outer_threads=best_outer, n_inner_threads=best_inner,
                        elapsed=elapsed, rss_delta_mb=rss_delta / 1024**2,
                    )
                    results.append(entry)

                    if verbose:
                        unrl_tag = 'on' if unrolling else 'off'
                        print(f'  {eng:6s} {sp:10s} {fmt:7s} '
                              f'unroll={unrl_tag} '
                              f'omp={best_outer} inner={best_inner} '
                              f'{elapsed:.3f}s  '
                              f'mem~{rss_delta/1024**2:.1f}MiB')

    results.sort(key=lambda x: x['elapsed'])
    if results:
        results[0]['recommended'] = True
    return results
