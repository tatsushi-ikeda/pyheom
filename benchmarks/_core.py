#  PyHEOM benchmark core -- system definition and shared utilities.
#  Based on examples/basic/brownian_oscillator_heom.py (2-level Brownian oscillator).

import time
import numpy as np
import pyheom.pylibheom as _lb
from pyheom import HEOMSolver, noise_decomposition, Brown

# ---------------------------------------------------------------------------
# System parameters (dimensionless, from brownian_oscillator_heom.py)
# ---------------------------------------------------------------------------
LAMBDA_0 = 0.01
OMEGA_0  = 1.0
ZETA     = 0.5
T        = 1.0
N_TIERS  = 5

# Benchmark run parameters (shorter than the example for fast iteration)
T_FINAL      = 5.0
DT_CALLBACK  = 0.25
DT           = 2.5e-3

# Per-solver default kwargs; rkdp additionally requires atol/rtol for its adaptive stepper.
SOLVER_KWARGS = {
    'lsrk4': {'dt': DT},
    'rk4':   {'dt': DT},
    'rkdp':  {'dt': DT, 'atol': 1e-8, 'rtol': 1e-6},
}


def solve_kwargs(solver):
    return SOLVER_KWARGS[solver]


def _H():
    omega_1 = np.sqrt(OMEGA_0**2 - ZETA**2 * 0.25)
    return np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)


def _corr():
    J = Brown(LAMBDA_0, ZETA, OMEGA_0)
    corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return corr


def _rho0():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def build_solver(engine, space, fmt, solver='lsrk4', **kwargs):
    """Build a HEOMSolver for the benchmark system.

    Returns None if the (engine, space, fmt) combination is not compiled.
    """
    try:
        return HEOMSolver(_H(), [_corr()], engine=engine, space=space,
                          format=fmt, solver=solver, n_tiers=N_TIERS,
                          n_inner_threads=1, **kwargs)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def run_trial(qme, solver='lsrk4', t_final=T_FINAL, dt_callback=DT_CALLBACK):
    """Run one solve call; return wall-clock seconds."""
    t_list = np.arange(0.0, t_final + dt_callback * 0.5, dt_callback)
    t0 = time.perf_counter()
    qme.solve(_rho0(), t_list, **solve_kwargs(solver))
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

ALL_ENGINES = ['eigen', 'mkl', 'cuda']
ALL_SPACES  = ['hilbert', 'liouville', 'ado']
ALL_FORMATS = ['dense', 'sparse']
ALL_SOLVERS = list(SOLVER_KWARGS)


def available_engines():
    return [e for e in ALL_ENGINES
            if getattr(_lb, f'{e}_is_supported', lambda: False)()]


def full_grid(engines=None):
    """All (engine, space, format, solver) tuples for the given engines."""
    if engines is None:
        engines = available_engines()
    return [(eng, sp, fmt, slv)
            for eng in engines
            for sp  in ALL_SPACES
            for fmt in ALL_FORMATS
            for slv in ALL_SOLVERS]
