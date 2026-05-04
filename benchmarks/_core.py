#  PyHEOM benchmark core -- system definition and shared utilities.

import time
import numpy as np
import pyheom.pylibheom as _lb
from pyheom import HEOMSolver, noise_decomposition, Brown

# ---------------------------------------------------------------------------
# System parameters -- match examples/basic/brownian_oscillator_heom.py
# ---------------------------------------------------------------------------
LAMBDA_0 = 0.1
OMEGA_0  = 1.0
ZETA     = 0.5
T        = 1.0
J_COUP   = 0.1   # off-diagonal coupling in H
N_TIERS  = 10

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
    return np.array([[omega_1, J_COUP], [J_COUP, 0.0]], dtype=np.complex128)


def _corr():
    J_sd = Brown(LAMBDA_0, ZETA, OMEGA_0)
    corr = noise_decomposition(J_sd, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return corr


def _rho0():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def build_solver(engine, space, fmt, solver='lsrk4', unrolling=True, **kwargs):
    """Build a HEOMSolver for the benchmark system.

    Returns None if the (engine, space, fmt) combination is not compiled.
    unrolling=False forces dynamic n_level_c regardless of engine.
    n_inner_threads and n_outer_threads can be passed via **kwargs.
    """
    try:
        return HEOMSolver(_H(), [_corr()], engine=engine, space=space,
                          format=fmt, solver=solver, n_tiers=N_TIERS,
                          unrolling=unrolling, **kwargs)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def run_trial(qme, t_final=T_FINAL, dt_callback=DT_CALLBACK):
    """Run one solve call; return wall-clock seconds."""
    t_list = np.arange(0.0, t_final + dt_callback * 0.5, dt_callback)
    t0 = time.perf_counter()
    qme.solve(_rho0(), t_list, **solve_kwargs(FIXED_SOLVER))
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

ALL_ENGINES    = ['Eigen', 'MKL', 'CUDA']
ALL_SPACES     = ['Hilbert', 'Liouville', 'ADO']
ALL_FORMATS    = ['dense', 'sparse']
ALL_UNROLLINGS = [True, False]
FIXED_SOLVER   = 'lsrk4'


def available_engines():
    return [e for e in ALL_ENGINES
            if getattr(_lb, f'{e.lower()}_is_supported', lambda: False)()]


def full_grid(engines=None, unrollings=None):
    """All (engine, space, format, unrolling) tuples for the given engines."""
    if engines is None:
        engines = available_engines()
    if unrollings is None:
        unrollings = [True]
    return [(eng, sp, fmt, unrl)
            for eng in engines
            for sp  in ALL_SPACES
            for fmt in ALL_FORMATS
            for unrl in unrollings]
