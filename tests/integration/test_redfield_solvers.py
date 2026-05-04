"""Integration test: rk4, lsrk4, rkdp solvers for RedfieldSolver.

Mirrors test_solvers.py but exercises RedfieldSolver (not HEOMSolver).
All three solvers applied to the same 2-level Brownian-oscillator system
must agree on populations to within their respective truncation error bounds.
"""

import numpy as np
import pytest

from pyheom import RedfieldSolver, noise_decomposition, Brown, Integrator

pytestmark = pytest.mark.integration

TARGET_T    = 4.975
CALLBACK_DT = 2.5e-2


def _build_solver(solver_name):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return RedfieldSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver=solver_name,
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


class TestRedfieldSolverAgreement:
    """All three ODE solvers must produce consistent results for Redfield."""

    def test_rk4_lsrk4_agree(self):
        # Same dt and order -- floating-point agreement
        rho_rk4   = _run('rk4',   dt=0.25e-2)
        rho_lsrk4 = _run('lsrk4', dt=0.25e-2)
        assert rho_rk4 == pytest.approx(rho_lsrk4, abs=1e-12)

    def test_lsrk4_rkdp_agree(self):
        # Adaptive rkdp with tight tolerance agrees with fixed-step lsrk4
        rho_lsrk4 = _run('lsrk4', dt=0.25e-2)
        rho_rkdp  = _run('rkdp',  dt=0.25e-2, atol=1e-8, rtol=1e-6)
        assert rho_lsrk4 == pytest.approx(rho_rkdp, abs=1e-10)

    def test_rk4_rkdp_agree(self):
        rho_rk4  = _run('rk4',  dt=0.25e-2)
        rho_rkdp = _run('rkdp', dt=0.25e-2, atol=1e-8, rtol=1e-6)
        assert rho_rk4 == pytest.approx(rho_rkdp, abs=1e-10)


class TestRedfieldRKDPAdaptivity:

    def test_rkdp_within_tolerance(self):
        # rkdp with atol=1e-8, rtol=1e-6 must agree with very fine lsrk4
        rho_ref  = _run('lsrk4', dt=0.25e-2 / 4)
        rho_rkdp = _run('rkdp', dt=0.25e-2, atol=1e-8, rtol=1e-6)
        assert abs(rho_rkdp - rho_ref) < 1e-6

    def test_rkdp_trace_conservation(self):
        """rkdp must conserve trace throughout the evolution."""
        qme = _build_solver('rkdp')
        rho_0 = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0
        traces = []
        t_list = np.arange(0.0, 5.0 + CALLBACK_DT * 0.5, CALLBACK_DT)

        def callback(t):
            traces.append(float((qme.rho[0, 0] + qme.rho[1, 1]).real))

        qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2, atol=1e-8, rtol=1e-6)

        for i, tr in enumerate(traces):
            assert tr == pytest.approx(1.0, abs=1e-10), \
                f'trace violated at step {i}: {tr}'

    def test_rkdp_e_ops(self):
        """e_ops must work with rkdp+RedfieldSolver and match manual trace."""
        t_list  = np.arange(0.0, 3.0 + CALLBACK_DT * 0.5, CALLBACK_DT)
        rho_0   = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0
        proj0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

        # rkdp via e_ops
        qme1 = _build_solver('rkdp')
        result = qme1.solve(rho_0, t_list, e_ops=[proj0],
                            dt=0.25e-2, atol=1e-8, rtol=1e-6)

        # lsrk4 via callback (reference)
        qme2 = _build_solver('lsrk4')
        manual = []
        def cb(t):
            manual.append(float(qme2.rho[0, 0].real))
        qme2.solve(rho_0, t_list, callback=cb, dt=0.25e-2)

        np.testing.assert_allclose(
            result.expect[0].real, manual, atol=1e-8,
        )


class TestRedfieldIntegrator:
    """Integrator interface (qme.init / advance_to) for RedfieldSolver."""

    def test_init_returns_integrator(self):
        qme = _build_solver('lsrk4')
        integrator = qme.init(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128), dt=1e-3)
        assert isinstance(integrator, Integrator)

    def test_integrator_matches_solve(self):
        """Step-by-step advance must give the same result as a single solve()."""
        t_final = 3.0
        t_list  = np.linspace(0.0, t_final, 301)
        rho_0   = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0

        qme_ref = _build_solver('lsrk4')
        qme_ref.solve(rho_0, t_list, dt=1e-3)
        rho_ref = qme_ref.rho.copy()

        qme_int    = _build_solver('lsrk4')
        integrator = qme_int.init(rho_0, dt=1e-3)
        for t in [1.0, 2.0, t_final]:
            integrator.advance_to(t)

        np.testing.assert_allclose(integrator.rho, rho_ref, atol=1e-8)

    def test_integrator_rkdp(self):
        """Integrator works with rkdp solver for RedfieldSolver."""
        rho_0 = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0

        qme_ref = _build_solver('lsrk4')
        t_list = np.linspace(0.0, 3.0, 301)
        qme_ref.solve(rho_0, t_list, dt=1e-3)
        rho_ref = qme_ref.rho.copy()

        qme_int    = _build_solver('rkdp')
        integrator = qme_int.init(rho_0, dt=1e-3, atol=1e-10, rtol=1e-8)
        integrator.advance_to(3.0)

        np.testing.assert_allclose(integrator.rho, rho_ref, atol=1e-6)


class TestRedfieldFixedStepConvergence:
    """rk4 and lsrk4 error decreases as dt is reduced."""

    @pytest.mark.parametrize('solver', ['rk4', 'lsrk4'])
    def test_smaller_dt_gives_smaller_error(self, solver):
        rho_ref    = _run('rkdp', dt=0.25e-2 / 4, atol=1e-12, rtol=1e-10)
        rho_fine   = _run(solver, dt=0.25e-2 / 2)
        rho_coarse = _run(solver, dt=0.25e-2)

        err_fine   = abs(rho_fine   - rho_ref)
        err_coarse = abs(rho_coarse - rho_ref)

        assert err_coarse < 1e-8, \
            f'{solver} coarse dt error too large: {err_coarse}'
        assert err_fine <= err_coarse * 2 + 1e-13, \
            f'{solver}: fine-dt error {err_fine:.2e} not <= coarse-dt error {err_coarse:.2e}'
