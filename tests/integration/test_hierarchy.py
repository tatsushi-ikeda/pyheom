"""Integration test: HEOM hierarchy size satisfies the combinatoric formula.

For K bath modes and N tiers, the number of auxiliary density operators
(ADOs) is n_hrchy = C(K+N, K) (stars-and-bars with |n| <= N).

ptr_p1/ptr_m1 access is not yet exposed through the Python bindings;
that check is deferred to the C++ unit tests (L_TEST5).
"""

import numpy as np
import pytest
from math import comb

from pyheom import HEOMSolver, noise_decomposition, Drude, Brown

pytestmark = pytest.mark.integration


def _make_solver(corr, truncation_depth):
    """Build a minimal HEOMSolver for hierarchy-structure queries."""
    H = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    corr.V = V

    return HEOMSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver='lsrk4',
        truncation_depth=truncation_depth, n_inner_threads=1, n_outer_threads=1,
    )


# ---------------------------------------------------------------------------
# n_hrchy = C(K+N, K)
# ---------------------------------------------------------------------------

class TestHierarchyCount:

    @pytest.fixture(params=[1, 2, 3, 4, 5])
    def truncation_depth(self, request):
        return request.param

    def test_single_drude_mode(self, truncation_depth):
        # K=1: single Drude pole, type_ltc='none'
        corr = noise_decomposition(Drude(eta=1.0, gamma_c=1.0), T=1.0, type_ltc='none')
        K = len(corr.phi_0)
        assert K == 1
        qme = _make_solver(corr, truncation_depth)
        assert qme.qme_impl.get_n_hierarchy() == comb(K + truncation_depth, K)

    def test_two_mode_overdamped_brown(self, truncation_depth):
        # K=2: overdamped Brown -> two real poles, type_ltc='none'
        corr = noise_decomposition(Brown(1.0, 4.0, 1.0), T=1.0, type_ltc='none')
        K = len(corr.phi_0)
        assert K == 2
        qme = _make_solver(corr, truncation_depth)
        assert qme.qme_impl.get_n_hierarchy() == comb(K + truncation_depth, K)

    def test_three_mode_underdamped_brown_psd(self, truncation_depth):
        # K=3: underdamped Brown + 1 PSD mode
        corr = noise_decomposition(
            Brown(0.01, 0.5, 1.0), T=1.0,
            type_ltc='psd', n_psd=1, type_psd='n-1/n'
        )
        K = len(corr.phi_0)
        assert K == 3
        qme = _make_solver(corr, truncation_depth)
        assert qme.qme_impl.get_n_hierarchy() == comb(K + truncation_depth, K)


class TestStorageSize:

    @pytest.mark.parametrize("truncation_depth", [1, 2, 3])
    def test_storage_size_is_n_hrchy_plus_one(self, truncation_depth):
        corr = noise_decomposition(Drude(eta=1.0, gamma_c=1.0), T=1.0, type_ltc='none')
        qme = _make_solver(corr, truncation_depth)
        n_hrchy = qme.qme_impl.get_n_hierarchy()
        assert qme.storage_size() == n_hrchy + 1

    @pytest.mark.parametrize("truncation_depth", [1, 2, 3])
    def test_rho_hierarchy_shape(self, truncation_depth):
        corr = noise_decomposition(Drude(eta=1.0, gamma_c=1.0), T=1.0, type_ltc='none')
        qme = _make_solver(corr, truncation_depth)
        rho_0 = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0
        qme.solve(rho_0, np.array([0.0, 0.1]), dt=0.01)
        assert qme.rho_hierarchy.shape == (qme.storage_size(), 2, 2)
        assert qme.rho.shape == (2, 2)

    @pytest.mark.parametrize("truncation_depth", [1, 2, 3])
    def test_rho_is_view_of_hierarchy(self, truncation_depth):
        corr = noise_decomposition(Drude(eta=1.0, gamma_c=1.0), T=1.0, type_ltc='none')
        qme = _make_solver(corr, truncation_depth)
        # rho property is _rho[0] -- a view, same memory
        rho_0 = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0
        qme.solve(rho_0, np.array([0.0, 0.1]), dt=0.01)
        assert np.shares_memory(qme.rho, qme.rho_hierarchy)
