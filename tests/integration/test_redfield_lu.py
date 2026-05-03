"""Integration test: Redfield LU decomposition with many Pade poles.

Uses n_psd=5 (K=7 modes, 7x7 gamma matrix per noise source) to exercise
the LU decomposition used in RedfieldSolver.correlation().  Serves as a
regression test for the pivot-selection fix in linalg_engine_nil.

Reference values generated with the corrected Eigen build.
"""

import numpy as np
import pytest

from pyheom import RedfieldSolver, noise_decomposition, Brown


REFERENCE = [
    (4.975,  0.5700776318141149),
    (9.975,  0.3946377775747397),
    (14.975, 0.3235880170536541),
    (24.975, 0.28316140496472986),
]
TOL = 1e-10

pytestmark = pytest.mark.integration


def _build_solver():
    lambda_0 = 0.01
    omega_0  = 1.0
    zeta     = 0.5
    T        = 1.0

    J = Brown(lambda_0, zeta, omega_0)
    # n_psd=5 -> K=7 modes, exercising the 7x7 LU path in correlation()
    corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=5, type_psd='n-1/n')

    omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    return RedfieldSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver='lsrk4',
    )


def test_redfield_lu_n_modes():
    """n_psd=5 with Brown bath must yield K=7 modes."""
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=5, type_psd='n-1/n')
    assert len(corr.phi_0) == 7


def test_redfield_lu_population_dynamics():
    qme = _build_solver()
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0

    target_times = [t for t, _ in REFERENCE]
    callback_dt  = 2.5e-2
    t_list = np.arange(0.0, target_times[-1] + callback_dt * 0.5, callback_dt)
    captured = {}

    def callback(t):
        for ref_t in target_times:
            if ref_t not in captured and abs(t - ref_t) < callback_dt * 0.51:
                captured[ref_t] = float(qme.rho[0, 0].real)

    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)

    for ref_t, ref_val in REFERENCE:
        assert ref_t in captured, f"callback missed t={ref_t}"
        assert captured[ref_t] == pytest.approx(ref_val, abs=TOL), \
            f"t={ref_t}: got {captured[ref_t]}, expected {ref_val}"


def test_redfield_lu_trace_conservation():
    qme = _build_solver()
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0

    traces = []
    t_list = np.arange(0.0, 5.0 + 2.5e-2 * 0.5, 2.5e-2)

    def callback(t):
        traces.append(float((qme.rho[0, 0] + qme.rho[1, 1]).real))

    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)

    for i, tr in enumerate(traces):
        assert tr == pytest.approx(1.0, abs=1e-12), \
            f"trace violated at step {i}: {tr}"
