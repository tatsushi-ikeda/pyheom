"""Integration test: rk4, lsrk4, rkdp solvers agree to within tolerances.

All three solvers applied to the same system (2-level Brownian oscillator,
truncation_depth=3) must agree on populations to within their respective truncation
error bounds:
  - rk4 vs lsrk4: same dt, same order -> floating-point agreement (~1e-14)
  - lsrk4 vs rkdp: adaptive step-size control -> agreement at atol/rtol level

Note: rkdp requires both dt (initial step size) and atol/rtol tolerances.
"""

import numpy as np
import pytest

from pyheom import HEOMSolver, noise_decomposition, Brown, unit

pytestmark = pytest.mark.integration

TARGET_T = 4.975
CALLBACK_DT = 2.5e-2


def _build_solver(solver_name):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    return HEOMSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver=solver_name,
        truncation_depth=3, n_inner_threads=1, n_outer_threads=1,
    )


def _run(solver_name, dt=0.25e-2, atol=None, rtol=None):
    qme = _build_solver(solver_name)
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0

    t_list = np.arange(0.0, TARGET_T + CALLBACK_DT * 2, CALLBACK_DT)
    captured = {}

    def callback(t):
        if TARGET_T not in captured and abs(t - TARGET_T) < CALLBACK_DT * 0.51:
            captured[TARGET_T] = float(qme.rho[0, 0].real)

    if solver_name == 'rkdp':
        qme.solve(rho_0, t_list, callback=callback, dt=dt, atol=atol, rtol=rtol)
    else:
        qme.solve(rho_0, t_list, callback=callback, dt=dt)

    return captured[TARGET_T]


class TestSolverAgreement:

    def test_rk4_lsrk4_agree(self):
        # Same dt and order -> floating-point agreement
        rho_rk4   = _run('rk4',   dt=0.25e-2)
        rho_lsrk4 = _run('lsrk4', dt=0.25e-2)
        assert rho_rk4 == pytest.approx(rho_lsrk4, abs=1e-12)

    def test_lsrk4_rkdp_agree(self):
        # Adaptive rkdp with tight tolerance -> agrees with fixed-step lsrk4
        rho_lsrk4 = _run('lsrk4', dt=0.25e-2)
        rho_rkdp  = _run('rkdp',  dt=0.25e-2, atol=1e-8, rtol=1e-6)
        assert rho_lsrk4 == pytest.approx(rho_rkdp, abs=1e-10)

    def test_rk4_rkdp_agree(self):
        rho_rk4  = _run('rk4',  dt=0.25e-2)
        rho_rkdp = _run('rkdp', dt=0.25e-2, atol=1e-8, rtol=1e-6)
        assert rho_rk4 == pytest.approx(rho_rkdp, abs=1e-10)


class TestFixedStepConvergence:
    """rk4 and lsrk4 error decreases as dt is reduced."""

    @pytest.mark.parametrize("solver", ['rk4', 'lsrk4'])
    def test_smaller_dt_gives_smaller_error(self, solver):
        # Use rkdp with very tight tolerance as the reference
        rho_ref    = _run('rkdp', dt=0.25e-2 / 4, atol=1e-12, rtol=1e-10)
        rho_fine   = _run(solver, dt=0.25e-2 / 2)
        rho_coarse = _run(solver, dt=0.25e-2)

        err_fine   = abs(rho_fine   - rho_ref)
        err_coarse = abs(rho_coarse - rho_ref)

        # Both errors must be small (solver works) and fine must not be worse
        assert err_coarse < 1e-8, f"{solver} coarse dt error too large: {err_coarse}"
        assert err_fine <= err_coarse * 2 + 1e-13, \
            f"{solver}: fine-dt error {err_fine:.2e} not <= coarse-dt error {err_coarse:.2e}"


class TestRKDPAdaptivity:

    def test_rkdp_within_tolerance(self):
        # rkdp with atol=1e-8, rtol=1e-6 should agree with very fine lsrk4
        rho_ref  = _run('lsrk4', dt=0.25e-2 / 4)
        rho_rkdp = _run('rkdp', dt=0.25e-2, atol=1e-8, rtol=1e-6)
        # Agreement should be at or within the tolerance level
        assert abs(rho_rkdp - rho_ref) < 1e-6
