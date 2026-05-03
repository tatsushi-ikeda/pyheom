"""Tests for pyheom.noise_decomposition: output structure and analytic values.

Focus on noise_decomposition(J, T, 'none') so that analytic values can be
derived from the spectral density poles alone.
"""

import numpy as np
import pytest

from pyheom.predefined_noise import Drude, Brown
from pyheom.noise_decomposition import (
    noise_decomposition, BathCorrelation, calc_noise_time_domain, calc_noise_params,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eval_a_from_noise(nd, t):
    """A(t) = sum_k (a_mat.T @ sigma)_k * phi_0_k * exp(-gamma_kk * t)."""
    sigma_a    = nd.a_mat.T.toarray() @ nd.sigma
    gamma_diag = nd.gamma.diagonal()
    return float(np.real(np.dot(sigma_a * nd.phi_0, np.exp(-gamma_diag * t))))


# ---------------------------------------------------------------------------
# Drude: single first-order pole
# ---------------------------------------------------------------------------

class TestDrudeNone:
    # Drude(eta=2, gamma_c=3), T=1
    # A(t) = -9*exp(-3t)   [from -(b/2)*exp(-at) with b=18, a=3]
    # S(t) = 6*exp(-3t)    [from (b*T/a)*exp(-at) with T=1]
    # phi_dim = 1

    def setup_method(self):
        self.nd = noise_decomposition(Drude(eta=2.0, gamma_c=3.0), T=1.0, type_ltc='none')

    def test_returns_bath_correlation(self):
        assert isinstance(self.nd, BathCorrelation)

    def test_phi_dim(self):
        assert self.nd.phi_0.shape == (1,)
        assert self.nd.sigma.shape == (1,)
        assert self.nd.gamma.shape == (1, 1)
        assert self.nd.s_mat.shape == (1, 1)
        assert self.nd.a_mat.shape == (1, 1)

    def test_phi_0(self):
        assert self.nd.phi_0[0] == pytest.approx(1.0)

    def test_sigma(self):
        assert self.nd.sigma[0] == pytest.approx(1.0)

    def test_gamma_diagonal(self):
        # gamma_c = 3.0 (single Drude pole)
        gamma_dense = self.nd.gamma.toarray()
        assert complex(gamma_dense[0, 0]).real == pytest.approx(3.0, rel=1e-12)
        assert complex(gamma_dense[0, 0]).imag == pytest.approx(0.0, abs=1e-12)

    def test_s_mat(self):
        # s_mat = [[S[(3,0)]/sigma[0]]] = [[6.0]]
        s_dense = self.nd.s_mat.toarray()
        assert complex(s_dense[0, 0]).real == pytest.approx(6.0, rel=1e-12)

    def test_a_mat(self):
        # a_mat = [[A[(3,0)]/sigma[0]]] = [[-9.0]]
        a_dense = self.nd.a_mat.toarray()
        assert complex(a_dense[0, 0]).real == pytest.approx(-9.0, rel=1e-12)

    def test_s_delta_zero(self):
        assert self.nd.s_delta == 0.0

    @pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
    def test_a_time_reconstruction(self, t):
        # A(t) = -9*exp(-3t) from noise params
        expected = -9.0 * np.exp(-3.0 * t)
        assert eval_a_from_noise(self.nd, t) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Brown overdamped: two uncoupled first-order real poles
# ---------------------------------------------------------------------------

class TestBrownOverdampedNone:
    # Brown(lambda_0=1, gamma_c=4, omega_0=1), T=1
    # gamma_p_real = 2-sqrt(3), gamma_m_real = 2+sqrt(3)
    # phi_dim = 2, diagonal matrices

    def setup_method(self):
        self.nd = noise_decomposition(
            Brown(lambda_0=1.0, gamma_c=4.0, omega_0=1.0), T=1.0, type_ltc='none'
        )
        self._gp = 2.0 - np.sqrt(3)   # 0.2679...
        self._gm = 2.0 + np.sqrt(3)   # 3.7320...

    def test_phi_dim(self):
        assert self.nd.phi_0.shape == (2,)
        assert self.nd.gamma.shape == (2, 2)
        assert self.nd.s_mat.shape == (2, 2)

    def test_phi_0_all_ones(self):
        assert self.nd.phi_0 == pytest.approx([1.0, 1.0], abs=1e-12)

    def test_sigma_all_ones(self):
        assert self.nd.sigma == pytest.approx([1.0, 1.0], abs=1e-12)

    def test_gamma_diagonal_values(self):
        gamma_diag = np.real(self.nd.gamma.diagonal())
        # sort so order-independent
        assert sorted(gamma_diag) == pytest.approx(
            sorted([self._gp, self._gm]), rel=1e-10
        )

    def test_s_delta_zero(self):
        assert self.nd.s_delta == 0.0

    def test_s_mat_diagonal(self):
        s_dense  = np.real(self.nd.s_mat.toarray())
        off_diag = s_dense - np.diag(np.diag(s_dense))
        assert np.max(np.abs(off_diag)) == pytest.approx(0.0, abs=1e-12)

    def test_a_mat_diagonal(self):
        a_dense  = np.real(self.nd.a_mat.toarray())
        off_diag = a_dense - np.diag(np.diag(a_dense))
        assert np.max(np.abs(off_diag)) == pytest.approx(0.0, abs=1e-12)

    def test_a_mat_antisymmetric_sum(self):
        # For overdamped Brown: A[(gp,0)] = -coef_p/2, A[(gm,0)] = -coef_m/2
        # coef_p = 1/sqrt(3), coef_m = -1/sqrt(3) -> A terms cancel in sum
        a_diag = np.real(self.nd.a_mat.diagonal())
        # a_mat sum = -1/(2sqrt(3)) + 1/(2sqrt(3)) = 0
        assert sum(a_diag) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
    def test_a_time_reconstruction(self, t):
        # A(t) from noise params should match direct calc_a_from_poles
        from pyheom.summation_over_poles import calc_a_from_poles
        poles    = Brown(lambda_0=1.0, gamma_c=4.0, omega_0=1.0).poles
        a_result = calc_a_from_poles(poles)
        expected = sum(c * t**l * np.exp(-a * t) for (a, l), c in a_result.items())
        got      = eval_a_from_noise(self.nd, t)
        assert got == pytest.approx(float(np.real(expected)), rel=1e-10)


# ---------------------------------------------------------------------------
# calc_noise_time_domain: type_ltc comparison
# ---------------------------------------------------------------------------

class TestCalcNoiseTimeDomainDrude:
    """Verify S and A pole dicts for Drude+'none' against analytic values."""

    def setup_method(self):
        self.J = Drude(eta=2.0, gamma_c=3.0)

    def test_a_single_key(self):
        _, A = calc_noise_time_domain(self.J, T=1.0, type_ltc='none')
        assert list(A.keys()) == [(3.0, 0)]

    def test_a_value(self):
        _, A = calc_noise_time_domain(self.J, T=1.0, type_ltc='none')
        assert float(A[(3.0, 0)]) == pytest.approx(-9.0, rel=1e-12)

    def test_s_single_key(self):
        S, _ = calc_noise_time_domain(self.J, T=1.0, type_ltc='none')
        assert list(S.keys()) == [(3.0, 0)]

    def test_s_value(self):
        S, _ = calc_noise_time_domain(self.J, T=1.0, type_ltc='none')
        assert float(S[(3.0, 0)]) == pytest.approx(6.0, rel=1e-12)

    def test_s_scales_with_temperature(self):
        # S is proportional to T
        S1, _ = calc_noise_time_domain(self.J, T=1.0, type_ltc='none')
        S2, _ = calc_noise_time_domain(self.J, T=2.0, type_ltc='none')
        assert float(S2[(3.0, 0)]) == pytest.approx(2 * float(S1[(3.0, 0)]), rel=1e-12)

    def test_a_independent_of_temperature(self):
        _, A1 = calc_noise_time_domain(self.J, T=1.0, type_ltc='none')
        _, A2 = calc_noise_time_domain(self.J, T=2.0, type_ltc='none')
        assert float(A1[(3.0, 0)]) == pytest.approx(float(A2[(3.0, 0)]), rel=1e-12)
