"""Tests for pyheom.pade_decomposition: Pade poles reproduce coth(omega/(2T))/2.

The PSD approximates coth(omega/(2T))/2 (= n(omega)+1/2) as:
    g(omega) = T/omega + R_1*omega + T_3*omega^3
               + sum_k 2*eta_k*T*omega / (xi_k^2*T^2 + omega^2)

Accuracy improves monotonically with n_psd.
"""

import numpy as np
import pytest

from pyheom.pade_decomposition import psd

pytestmark = pytest.mark.unit

T_TEST = 1.0
OMEGA_RANGE = np.linspace(0.5, 5.0, 50)


def coth_half(omega, T):
    """Exact coth(omega/(2T))/2."""
    return 0.5 / np.tanh(omega / (2 * T))


def psd_approx(xi, eta, R_1, T_3, T, omega):
    """Evaluate the PSD approximation of coth(omega/(2T))/2."""
    val = T / omega + float(R_1) * omega + float(T_3) * omega**3
    for xi_k, eta_k in zip(xi, eta):
        val += 2 * float(eta_k) * T * omega / (float(xi_k)**2 * T**2 + omega**2)
    return val


def max_error(xi, eta, R_1, T_3, T, omegas):
    approx = np.array([psd_approx(xi, eta, R_1, T_3, T, w) for w in omegas])
    exact  = coth_half(omegas, T)
    return np.max(np.abs(approx - exact))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestPSDStructure:

    @pytest.mark.parametrize("n_psd", [1, 2, 3, 5])
    def test_nn_output_lengths(self, n_psd):
        xi, eta, R_1, T_3 = psd(n_psd, 'n/n')
        assert len(xi)  == n_psd
        assert len(eta) == n_psd

    @pytest.mark.parametrize("n_psd", [1, 2, 3])
    def test_nm1_output_lengths(self, n_psd):
        xi, eta, R_1, T_3 = psd(n_psd, 'n-1/n')
        assert len(xi)  == n_psd
        assert len(eta) == n_psd

    @pytest.mark.parametrize("n_psd", [1, 2, 3])
    def test_np1_output_lengths(self, n_psd):
        xi, eta, R_1, T_3 = psd(n_psd, 'n+1/n')
        assert len(xi)  == n_psd
        assert len(eta) == n_psd

    @pytest.mark.parametrize("n_psd", [1, 2, 3])
    def test_xi_positive(self, n_psd):
        for type_psd in ('n/n', 'n-1/n', 'n+1/n'):
            xi, _, _, _ = psd(n_psd, type_psd)
            assert np.all(np.real(xi) > 0), f'{type_psd} n={n_psd}: negative xi'

    def test_nn_residue_positive(self):
        # For N/N, R_1 > 0 and T_3 == 0
        for n_psd in [1, 2, 3]:
            _, _, R_1, T_3 = psd(n_psd, 'n/n')
            assert R_1 > 0
            assert T_3 == 0.0

    def test_nm1_residues_zero(self):
        # For (N-1)/N, R_1 == 0 and T_3 == 0
        for n_psd in [1, 2, 3]:
            _, _, R_1, T_3 = psd(n_psd, 'n-1/n')
            assert R_1 == 0.0
            assert T_3 == 0.0


# ---------------------------------------------------------------------------
# Approximation quality (N/N)
# ---------------------------------------------------------------------------

class TestPSDAccuracyNN:
    """N/N Pade: accuracy improves rapidly with n_psd."""

    def test_n1_tolerance(self):
        xi, eta, R_1, T_3 = psd(1, 'n/n')
        assert max_error(xi, eta, R_1, T_3, T_TEST, OMEGA_RANGE) < 2e-3

    def test_n3_tolerance(self):
        xi, eta, R_1, T_3 = psd(3, 'n/n')
        assert max_error(xi, eta, R_1, T_3, T_TEST, OMEGA_RANGE) < 1e-7

    def test_n5_tolerance(self):
        xi, eta, R_1, T_3 = psd(5, 'n/n')
        assert max_error(xi, eta, R_1, T_3, T_TEST, OMEGA_RANGE) < 1e-10

    def test_monotone_improvement(self):
        errors = []
        for n_psd in [1, 2, 3, 4, 5]:
            xi, eta, R_1, T_3 = psd(n_psd, 'n/n')
            errors.append(max_error(xi, eta, R_1, T_3, T_TEST, OMEGA_RANGE))
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i], (
                f'n/n error not monotone: n={i+1} err={errors[i]:.2e}, '
                f'n={i+2} err={errors[i+1]:.2e}'
            )


# ---------------------------------------------------------------------------
# Approximation quality ((N-1)/N)
# ---------------------------------------------------------------------------

class TestPSDAccuracyNm1:

    def test_n3_tolerance(self):
        xi, eta, R_1, T_3 = psd(3, 'n-1/n')
        assert max_error(xi, eta, R_1, T_3, T_TEST, OMEGA_RANGE) < 1e-6

    def test_monotone_improvement(self):
        errors = []
        for n_psd in [1, 2, 3]:
            xi, eta, R_1, T_3 = psd(n_psd, 'n-1/n')
            errors.append(max_error(xi, eta, R_1, T_3, T_TEST, OMEGA_RANGE))
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i]


# ---------------------------------------------------------------------------
# Pinned values (regression)
# ---------------------------------------------------------------------------

class TestPSDPinnedValues:

    def test_nn_n1_xi(self):
        xi, eta, R_1, T_3 = psd(1, 'n/n')
        assert float(xi[0]) == pytest.approx(6.4807407, rel=1e-6)

    def test_nn_n1_residue(self):
        _, _, R_1, _ = psd(1, 'n/n')
        # R = 1/(4*(1+1)*b[1]) = 1/(4*2*5) = 1/40
        assert R_1 == pytest.approx(0.025, rel=1e-10)

    def test_nn_n0_returns_unit_residue(self):
        xi, eta, R_1, T_3 = psd(0, 'n/n')
        assert len(xi) == 0
        assert R_1 == pytest.approx(1 / 12, rel=1e-10)
