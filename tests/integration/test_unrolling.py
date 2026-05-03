"""Integration tests: unrolling parameter (static vs dynamic n_level_c template).

When unrolling=True + engine='eigen' + n_level in [2, 3, 4], libheom
instantiates a static template (c_level='2'/'3'/'4'); otherwise it uses the
dynamic template (c_level='n').  Both templates must produce identical results.
"""

import numpy as np
import pytest
import pyheom.pylibheom as _lb

from pyheom import HEOMSolver, RedfieldSolver, noise_decomposition, Brown


pytestmark = pytest.mark.integration

STATIC_LEVELS = [2, 3, 4]


# ---------------------------------------------------------------------------
# System builders for various n_level
# ---------------------------------------------------------------------------

def _make_H(n):
    """Diagonal Hamiltonian: equally spaced levels 0, 1, ..., n-1."""
    return np.diag(np.arange(n, dtype=np.complex128))


def _make_corr(n):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    V = np.zeros((n, n), dtype=np.complex128)
    for i in range(n - 1):
        V[i, i + 1] = V[i + 1, i] = 1.0
    corr.V = V
    return corr


def _rho0(n):
    rho = np.zeros((n, n), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def _run(cls, n, unrolling, **extra):
    H    = _make_H(n)
    corr = _make_corr(n)
    qme  = cls(H, [corr], engine='eigen', space='hilbert', format='dense',
               solver='lsrk4', unrolling=unrolling, **extra)
    t_list = np.arange(0.0, 1.0 + 0.25 * 0.5, 0.25)
    return qme.solve(_rho0(n), t_list, e_ops=[np.eye(n)], dt=2.5e-3)


# ---------------------------------------------------------------------------
# c_level selection tests
# ---------------------------------------------------------------------------

class TestConfigSelection:
    """get_config() selects c_level=str(n) for static levels, 'n' otherwise."""

    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_heom_static_level_selected_when_unrolling_on(self, n_level):
        H = _make_H(n_level)
        qme = HEOMSolver(H, [_make_corr(n_level)], engine='eigen',
                         unrolling=True, n_tiers=2)
        assert qme.config['c_level'] == str(n_level)

    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_heom_dynamic_level_selected_when_unrolling_off(self, n_level):
        H = _make_H(n_level)
        qme = HEOMSolver(H, [_make_corr(n_level)], engine='eigen',
                         unrolling=False, n_tiers=2)
        assert qme.config['c_level'] == 'n'

    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_redfield_static_level_selected_when_unrolling_on(self, n_level):
        H = _make_H(n_level)
        qme = RedfieldSolver(H, [_make_corr(n_level)], engine='eigen',
                             unrolling=True)
        assert qme.config['c_level'] == str(n_level)

    def test_n_level_5_always_dynamic(self):
        H = _make_H(5)
        for unrolling in [True, False]:
            qme = HEOMSolver(H, [_make_corr(5)], engine='eigen',
                             unrolling=unrolling, n_tiers=1)
            assert qme.config['c_level'] == 'n'

    @pytest.mark.skipif(not _lb.mkl_is_supported(), reason='MKL not available')
    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_mkl_always_dynamic(self, n_level):
        H = _make_H(n_level)
        qme = HEOMSolver(H, [_make_corr(n_level)], engine='mkl',
                         unrolling=True, n_tiers=2)
        assert qme.config['c_level'] == 'n'

    @pytest.mark.skipif(not _lb.cuda_is_supported(), reason='CUDA not available')
    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_cuda_always_dynamic(self, n_level):
        H = _make_H(n_level)
        qme = HEOMSolver(H, [_make_corr(n_level)], engine='cuda',
                         unrolling=True, n_tiers=2)
        assert qme.config['c_level'] == 'n'


# ---------------------------------------------------------------------------
# Numerical equivalence tests
# ---------------------------------------------------------------------------

class TestUnrollingEquivalence:
    """unrolling=True and unrolling=False must give identical results."""

    TOL = 1e-12

    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_heom_hilbert_dense(self, n_level):
        r_on  = _run(HEOMSolver, n_level, unrolling=True,  n_tiers=2)
        r_off = _run(HEOMSolver, n_level, unrolling=False, n_tiers=2)
        np.testing.assert_allclose(r_on.expect, r_off.expect, atol=self.TOL,
                                   err_msg=f'HEOM n={n_level}: unrolling mismatch')

    @pytest.mark.parametrize('n_level', STATIC_LEVELS)
    def test_redfield_hilbert_dense(self, n_level):
        r_on  = _run(RedfieldSolver, n_level, unrolling=True)
        r_off = _run(RedfieldSolver, n_level, unrolling=False)
        np.testing.assert_allclose(r_on.expect, r_off.expect, atol=self.TOL,
                                   err_msg=f'Redfield n={n_level}: unrolling mismatch')

    def test_heom_population_unrolling_on_2level(self):
        """Regression: unrolling=True n=2 matches the pinned HEOM reference."""
        REFERENCE = 0.7569977275059762

        H    = _make_H(2)
        corr = _make_corr(2)
        # use the exact same system as test_brownian_2level.py
        omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
        H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
        corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        qme = HEOMSolver(H, [corr], engine='eigen', space='liouville',
                         format='dense', solver='lsrk4', unrolling=True, n_tiers=5)
        t_list = np.arange(0.0, 5.0 + 2.5e-2 * 0.5, 2.5e-2)
        captured = {}

        def cb(t):
            if 4.975 not in captured and abs(t - 4.975) < 2.5e-2 * 0.51:
                captured[4.975] = float(qme.rho[0, 0].real)

        qme.solve(_rho0(2), t_list, callback=cb, dt=2.5e-3)
        assert captured[4.975] == pytest.approx(REFERENCE, abs=1e-10)

    def test_heom_population_unrolling_off_matches_on_2level(self):
        """unrolling=False gives same result as unrolling=True for n=2."""
        omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
        H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
        J = Brown(0.01, 0.5, 1.0)
        corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
        corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        def make(unrolling):
            return HEOMSolver(H, [corr], engine='eigen', space='liouville',
                              format='dense', solver='lsrk4',
                              unrolling=unrolling, n_tiers=5)

        qme_on  = make(True)
        qme_off = make(False)
        t_list  = np.arange(0.0, 5.0 + 2.5e-2 * 0.5, 2.5e-2)

        on_vals, off_vals = [], []
        qme_on.solve( _rho0(2), t_list, callback=lambda t: on_vals.append(float(qme_on.rho[0,0].real)),  dt=2.5e-3)
        qme_off.solve(_rho0(2), t_list, callback=lambda t: off_vals.append(float(qme_off.rho[0,0].real)), dt=2.5e-3)

        np.testing.assert_allclose(on_vals, off_vals, atol=1e-12)
