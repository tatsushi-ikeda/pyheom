#!/usr/bin/env python
#  PyHEOM benchmark sweep script.
#
#  Usage (from pyheom/ project root):
#
#    python benchmarks/sweep.py                           # default grid
#    python benchmarks/sweep.py --auto                    # auto-select + thread tuning
#    python benchmarks/sweep.py --engines eigen           # single engine
#    python benchmarks/sweep.py --spaces liouville ado    # specific spaces
#    python benchmarks/sweep.py --n-trials 5 --t-final 25.0
#    python benchmarks/sweep.py --output results.json
#
#  Install pytest-benchmark for statistical benchmarks:
#    pip install pytest-benchmark
#    pytest benchmarks/test_bench.py --benchmark-only -v

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Ensure the project root (containing pyheom/) is importable.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from benchmarks._core import (
    available_engines, full_grid, build_solver, run_trial,
    ALL_ENGINES, ALL_SPACES, ALL_FORMATS, ALL_UNROLLINGS,
    T_FINAL, DT_CALLBACK,
)
from benchmarks._auto import auto_select


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COLS = ('engine', 'space', 'format', 'unroll', 'omp', 'inner', 'time(s)', 'mem(MiB)')
_WIDTHS = (8, 11, 7, 7, 5, 6, 10, 10)
_FMT_HDR = '  '.join(f'{{:<{w}}}' for w in _WIDTHS)
_SEP = '-' * (sum(_WIDTHS) + 2 * (len(_WIDTHS) - 1))


def _row(r):
    tag = ' *' if r.get('recommended') else '  '
    return tag + _FMT_HDR.format(
        r['engine'], r['space'], r['format'],
        'on' if r.get('unrolling', True) else 'off',
        str(r.get('n_outer_threads', '-')),
        str(r.get('n_inner_threads', '-')),
        f"{r['elapsed']:.3f}",
        f"{r.get('rss_delta_mb', 0.0):.1f}",
    )


def print_table(results):
    print()
    print('  ' + _FMT_HDR.format(*_COLS))
    print('  ' + _SEP)
    for r in results:
        print(_row(r))
    print('  ' + _SEP)
    if any(r.get('recommended') for r in results):
        print('  * recommended')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='pyheom benchmark sweep -- engine/parameter grid timing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--auto', action='store_true',
        help='discover available engines, estimate memory, warmup, tune threads',
    )
    parser.add_argument('--engines', nargs='+', choices=ALL_ENGINES,
                        metavar='ENGINE')
    parser.add_argument('--spaces',  nargs='+', choices=ALL_SPACES,
                        metavar='SPACE')
    parser.add_argument('--formats', nargs='+', choices=ALL_FORMATS,
                        metavar='FORMAT')
    parser.add_argument('--unrollings', nargs='+', choices=['on', 'off'],
                        metavar='UNROLL',
                        help='static template unrolling: on, off, or both (default: on)')
    parser.add_argument('--n-outer-threads', nargs='+', type=int, metavar='N',
                        help='OMP outer-loop thread counts to sweep '
                             '(default: 1; specify to enable node-level OMP parallelism)')
    parser.add_argument('--n-inner-threads', nargs='+', type=int, metavar='N',
                        help='inner matrix-op thread counts to sweep; '
                             'sets n_inner_threads (Eigen::setNbThreads / '
                             'mkl_set_num_threads) (default: OMP_NUM_THREADS or cpu_count)')
    parser.add_argument('--n-trials', type=int, default=3,
                        help='timing trials per combination (default: 3)')
    parser.add_argument('--t-final', type=float, default=T_FINAL,
                        help=f'simulation end time (default: {T_FINAL})')
    parser.add_argument('--output', metavar='FILE',
                        help='save results as JSON')
    args = parser.parse_args()

    unrollings = None
    if args.unrollings:
        unrollings = [u == 'on' for u in args.unrollings]

    if args.auto:
        results = auto_select(
            engines=args.engines or available_engines(),
            spaces=args.spaces   or ALL_SPACES,
            formats=args.formats or ALL_FORMATS,
            unrollings=unrollings or [True],
            n_trials=args.n_trials, verbose=True,
        )
    else:
        engines    = args.engines or available_engines()
        unrollings = unrollings or [True]
        omp_list   = args.n_outer_threads  # None or list of ints
        inner_list = args.n_inner_threads  # None or list of ints
        # Build Cartesian product of (n_outer, n_inner) thread pairs.
        # When neither flag is given, use a single (None, None) pair so
        # the solver falls back to its optional_args defaults.
        if omp_list is None and inner_list is None:
            thread_pairs = [(None, None)]
        else:
            thread_pairs = [
                (no, ni)
                for no in (omp_list  or [None])
                for ni in (inner_list or [1])
            ]

        grid = [
            (eng, sp, fmt, unrl, no, ni)
            for eng  in engines
            for sp   in (args.spaces  or ALL_SPACES)
            for fmt  in (args.formats or ALL_FORMATS)
            for unrl in unrollings
            for no, ni in thread_pairs
        ]

        results = []
        for engine, space, fmt, unrolling, n_outer, n_inner in grid:
            solver_kw = {}
            if n_outer is not None:
                solver_kw['n_outer_threads'] = n_outer
            if n_inner is not None:
                solver_kw['n_inner_threads'] = n_inner
            qme = build_solver(engine, space, fmt, unrolling=unrolling, **solver_kw)
            if qme is None:
                continue
            times = [run_trial(qme, t_final=args.t_final)
                     for _ in range(args.n_trials)]
            elapsed = float(np.median(times))
            unrl_tag  = 'on' if unrolling else 'off'
            outer_tag = str(n_outer) if n_outer is not None else '-'
            inner_tag = str(n_inner) if n_inner is not None else '-'
            results.append(dict(engine=engine, space=space, format=fmt,
                                unrolling=unrolling, elapsed=elapsed,
                                n_outer_threads=n_outer,
                                n_inner_threads=n_inner))
            print(f'  {engine:6s} {space:10s} {fmt:7s} '
                  f'unroll={unrl_tag} omp={outer_tag:<3s} inner={inner_tag:<3s} '
                  f'{elapsed:.3f}s',
                  flush=True)

        results.sort(key=lambda x: x['elapsed'])

    print_table(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
