"""pytest-benchmark tests for pyheom HEOMSolver.

Run:
    pytest benchmarks/test_bench.py --benchmark-only -v
    pytest benchmarks/test_bench.py --benchmark-only --benchmark-save=baseline
    pytest benchmarks/test_bench.py --benchmark-only --benchmark-compare=baseline

Requires:  pip install pytest-benchmark
"""
pytest_plugins = ()  # ensure conftest.py is loaded

import pytest
import numpy as np

pytest.importorskip('pytest_benchmark', reason='pytest-benchmark not installed; '
                    'run: pip install pytest-benchmark')

from ._core import (
    available_engines, build_solver, solve_kwargs,
    ALL_SPACES, ALL_FORMATS, ALL_UNROLLINGS, FIXED_SOLVER,
    T_FINAL, DT_CALLBACK, _rho0,
)


def _params():
    params = []
    for engine in available_engines():
        for space in ALL_SPACES:
            for fmt in ALL_FORMATS:
                for unrolling in ALL_UNROLLINGS:
                    unrl_tag = 'unroll=on' if unrolling else 'unroll=off'
                    params.append(pytest.param(
                        engine, space, fmt, unrolling,
                        id=f'{engine}/{space}/{fmt}/{unrl_tag}',
                    ))
    return params


@pytest.mark.parametrize('engine,space,fmt,unrolling', _params())
def test_heom_solve(benchmark, engine, space, fmt, unrolling):
    """Time HEOMSolver.solve() across all engine/space/format/unrolling combos."""
    qme = build_solver(engine, space, fmt, unrolling=unrolling)
    if qme is None:
        pytest.skip(f'not compiled: {engine}/{space}/{fmt}')

    t_list = np.arange(0.0, T_FINAL + DT_CALLBACK * 0.5, DT_CALLBACK)
    rho_0  = _rho0()

    benchmark(qme.solve, rho_0, t_list, **solve_kwargs(FIXED_SOLVER))
