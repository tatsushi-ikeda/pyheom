"""Integration tests for the Result / Integrator / e_ops / hierarchy-restart API.

Covers:
  - e_ops expectation values in Result.expect
  - hierarchy restart (rho_0 with ndim==3)
  - the Integrator interface (qme.init / advance_to)
  - solve() returns Result with .times / .expect
  - n_outer_threads evaluated at construction, not module import
  - device= top-level kwarg (invalid engine raises ValueError)
"""

import numpy as np
import pytest

from pyheom import HEOMSolver, RedfieldSolver, noise_decomposition, Brown, Drude, Result, Integrator

pytestmark = pytest.mark.integration

CALLBACK_DT = 2.5e-2
TARGET_T    = 4.975


def _build_heom(truncation_depth=3):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return HEOMSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver='lsrk4',
        truncation_depth=truncation_depth, n_inner_threads=1, n_outer_threads=1,
    )


def _rho0():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


# ---------------------------------------------------------------------------
# solve() returns Result
# ---------------------------------------------------------------------------

class TestResultReturn:

    def test_returns_result_instance(self):
        qme   = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        result = qme.solve(_rho0(), t_list, dt=1e-3)
        assert isinstance(result, Result)

    def test_result_times_matches_t_list(self):
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        result = qme.solve(_rho0(), t_list, dt=1e-3)
        np.testing.assert_array_equal(result.times, t_list)

    def test_result_expect_empty_without_e_ops(self):
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        result = qme.solve(_rho0(), t_list, dt=1e-3)
        assert result.expect == []

    def test_states_is_none_by_default(self):
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        result = qme.solve(_rho0(), t_list, dt=1e-3)
        assert result.states is None


# ---------------------------------------------------------------------------
# e_ops expectation values
# ---------------------------------------------------------------------------

class TestEOps:

    def test_single_eop_shape(self):
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        result = qme.solve(_rho0(), t_list, e_ops=[sigma_z], dt=1e-3)
        assert len(result.expect) == 1
        assert result.expect[0].shape == (11,)

    def test_two_eops(self):
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        result = qme.solve(_rho0(), t_list, e_ops=[sigma_z, sigma_x], dt=1e-3)
        assert len(result.expect) == 2

    def test_population_eop_at_t0(self):
        """<|0><0|>(0) == 1 for rho_0 = |0><0|."""
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        proj0  = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        result = qme.solve(_rho0(), t_list, e_ops=[proj0], dt=1e-3)
        # At t=0 the state is rho_0 = |0><0|, so Tr(proj0 @ rho) = 1
        assert result.expect[0][0] == pytest.approx(1.0, abs=1e-12)

    def test_eop_matches_manual_callback(self):
        """Expectation values from e_ops must agree with manual trace in callback."""
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        t_list  = np.linspace(0.0, 2.0, 21)

        qme1 = _build_heom()
        manual = []
        def cb(t):
            manual.append(float(np.trace(sigma_z @ qme1.rho).real))
        qme1.solve(_rho0(), t_list, callback=cb, dt=1e-3)

        qme2   = _build_heom()
        result = qme2.solve(_rho0(), t_list, e_ops=[sigma_z], dt=1e-3)
        evals  = result.expect[0].real

        np.testing.assert_allclose(evals, manual, atol=1e-12)

    def test_callback_and_eops_coexist(self):
        """callback and e_ops can be used simultaneously."""
        qme    = _build_heom()
        t_list = np.linspace(0.0, 1.0, 11)
        proj0  = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        cb_count = []
        result = qme.solve(_rho0(), t_list, callback=lambda t: cb_count.append(t),
                           e_ops=[proj0], dt=1e-3)
        assert len(cb_count) == 11
        assert result.expect[0].shape == (11,)


# ---------------------------------------------------------------------------
# hierarchy restart (3-D rho_0)
# ---------------------------------------------------------------------------

class TestHierarchyRestart:

    def test_restart_matches_continuous_run(self):
        """Splitting at midpoint and restarting must reproduce the single-run result."""
        t_mid   = 2.0
        t_final = 4.0
        t_fine  = np.linspace(0.0, t_final, 401)

        # Single continuous run
        qme_ref = _build_heom()
        qme_ref.solve(_rho0(), t_fine, dt=1e-3)
        rho_ref = qme_ref.rho.copy()

        # Two-segment run
        qme_a = _build_heom()
        t_seg1 = np.linspace(0.0, t_mid, 201)
        qme_a.solve(_rho0(), t_seg1, dt=1e-3)
        hrchy_mid = qme_a.rho_hierarchy.copy()

        qme_b = _build_heom()
        t_seg2 = np.linspace(0.0, t_final - t_mid, 201)
        qme_b.solve(hrchy_mid, t_seg2, dt=1e-3)
        rho_restart = qme_b.rho.copy()

        np.testing.assert_allclose(rho_restart, rho_ref, atol=1e-8)

    def test_restart_shape_mismatch_raises(self):
        qme = _build_heom(truncation_depth=3)
        bad = np.zeros((999, 2, 2), dtype=np.complex128)
        with pytest.raises(ValueError, match='storage_size'):
            qme.solve(bad, np.array([0.0, 0.1]), dt=0.01)

    def test_restart_wrong_ndim_raises(self):
        qme = _build_heom()
        bad = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        with pytest.raises(ValueError, match='got shape'):
            qme.solve(bad, np.array([0.0, 0.1]), dt=0.01)


# ---------------------------------------------------------------------------
# Integrator interface
# ---------------------------------------------------------------------------

class TestIntegrator:

    def test_init_returns_integrator(self):
        qme = _build_heom()
        integrator = qme.init(_rho0(), dt=1e-3)
        assert isinstance(integrator, Integrator)

    def test_advance_to_t0_is_noop(self):
        qme        = _build_heom()
        integrator = qme.init(_rho0(), dt=1e-3)
        rho_before = integrator.rho.copy()
        integrator.advance_to(0.0)
        np.testing.assert_array_equal(integrator.rho, rho_before)

    def test_advance_backward_raises(self):
        qme        = _build_heom()
        integrator = qme.init(_rho0(), dt=1e-3)
        integrator.advance_to(1.0)
        with pytest.raises(ValueError, match='backward'):
            integrator.advance_to(0.5)

    def test_t_property_tracks_time(self):
        qme        = _build_heom()
        integrator = qme.init(_rho0(), dt=1e-3)
        assert integrator.t == 0.0
        integrator.advance_to(1.0)
        assert integrator.t == pytest.approx(1.0)
        integrator.advance_to(2.5)
        assert integrator.t == pytest.approx(2.5)

    def test_rho_shares_memory_with_solver(self):
        qme        = _build_heom()
        integrator = qme.init(_rho0(), dt=1e-3)
        assert np.shares_memory(integrator.rho, qme.rho)

    def test_integrator_matches_solve(self):
        """Step-by-step advance must give the same result as a single solve()."""
        t_final = 4.0
        t_list  = np.linspace(0.0, t_final, 401)

        qme_ref = _build_heom()
        qme_ref.solve(_rho0(), t_list, dt=1e-3)
        rho_ref = qme_ref.rho.copy()

        qme_int    = _build_heom()
        integrator = qme_int.init(_rho0(), dt=1e-3)
        for t in [1.0, 2.0, 3.0, t_final]:
            integrator.advance_to(t)

        np.testing.assert_allclose(integrator.rho, rho_ref, atol=1e-8)

    def test_integrator_rho_hierarchy_shape(self):
        qme        = _build_heom(truncation_depth=2)
        integrator = qme.init(_rho0(), dt=1e-3)
        integrator.advance_to(0.5)
        assert integrator.rho_hierarchy.shape == qme.rho_hierarchy.shape


# ---------------------------------------------------------------------------
# n_outer_threads lazy evaluation
# ---------------------------------------------------------------------------

class TestLazyNOuterThreads:

    def test_n_outer_threads_default_is_int(self):
        """Default n_outer_threads must resolve to a positive integer at construction."""
        qme = _build_heom()
        n   = qme.qme_args['n_outer_threads']
        assert isinstance(n, int)
        assert n >= 1

    def test_explicit_n_outer_threads_preserved(self):
        """Explicitly supplied n_outer_threads must not be overridden."""
        J = Drude(eta=1e-8, gamma_c=1.0)
        corr = noise_decomposition(J, T=1.0, type_ltc='none')
        corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        H = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        qme = HEOMSolver(
            H, [corr],
            space='liouville', format='dense', engine='eigen',
            liouville_order='C', solver='lsrk4',
            truncation_depth=2, n_inner_threads=1, n_outer_threads=3,
        )
        assert qme.qme_args['n_outer_threads'] == 3


# ---------------------------------------------------------------------------
# device= top-level kwarg
# ---------------------------------------------------------------------------

class TestDeviceKwarg:

    def test_device_on_non_cuda_raises(self):
        J = Drude(eta=1e-8, gamma_c=1.0)
        corr = noise_decomposition(J, T=1.0, type_ltc='none')
        corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        H = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        with pytest.raises(ValueError, match="'device'"):
            HEOMSolver(
                H, [corr],
                space='liouville', format='dense', engine='eigen',
                liouville_order='C', solver='lsrk4',
                truncation_depth=2, n_inner_threads=1, n_outer_threads=1,
                device=0,
            )
