"""Value tests for the LTC paths of calc_bath_corr_poles: 'msd', 'psd', 'psd+fsd'.

The 'none' path is covered in test_noise_decomposition.py.  Here the symmetric
correlation S(t) reconstructed from the returned pole dict is checked against
the exact Drude correlation -- its Matsubara series summed to convergence --
and against analytic coefficient formulas.  The antisymmetric part A(t) is
temperature- and LTC-independent, so it must equal the 'none' result.
"""

import numpy as np
import pytest

from pyheom.spectral_density import Drude
from pyheom.noise_decomposition import calc_bath_corr_poles, calc_s_msd
from pyheom.summation_over_poles import calc_a_from_poles

pytestmark = pytest.mark.unit


ETA, GAMMA_C = 2.0, 3.0          # Drude(eta, gamma_c); A[(gamma_c,0)] = -eta gc^2/2 = -9
T_TEST = 1.0
T_GRID = np.array([0.3, 0.7, 1.5])


def matsubara_freqs(n, T):
    k = np.arange(1, n + 1)
    return 2 * np.pi * k * T


def S_exact_drude(eta, gc, T, t, n_terms=20000):
    """Exact symmetric Drude correlation S(t) for t>0, via its Matsubara series.

    S(t) = (eta gc^2/2) cot(gc/2T) e^{-gc t}
           + sum_k 2T eta gc^2 nu_k/(nu_k^2 - gc^2) e^{-nu_k t},  nu_k = 2 pi k T.
    The e^{-nu_k t} factor converges to machine precision in a few thousand
    terms for t bounded away from 0.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    cot = 1.0 / np.tan(gc / (2 * T))
    out = (eta * gc**2 / 2.0) * cot * np.exp(-gc * t)
    nu = matsubara_freqs(n_terms, T)
    coef = 2 * T * eta * gc**2 * nu / (nu**2 - gc**2)
    out = out + np.sum(coef[:, None] * np.exp(-nu[:, None] * t[None, :]), axis=0)
    return out


def reconstruct_S(S, t):
    """S(t) for t>0 from the pole dict (the delta term at a=inf is skipped)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros(len(t), dtype=np.complex128)
    for (a, l), c in S.items():
        if a == np.inf:
            continue
        out += c * t**l * np.exp(-a * t)
    return out.real


def coeff_at(S, freq, l=0, tol=1e-9):
    """Sum the coefficients of order-l poles whose frequency matches `freq`."""
    total = 0.0 + 0.0j
    for (a, ll), c in S.items():
        if a == np.inf or ll != l:
            continue
        a = complex(a)
        if abs(a.real - freq) < tol and abs(a.imag) < tol:
            total += complex(c)
    return total


# ---------------------------------------------------------------------------
# msd (Matsubara spectral decomposition)
# ---------------------------------------------------------------------------

class TestMSD:
    def setup_method(self):
        self.J = Drude(eta=ETA, gamma_c=GAMMA_C)
        self.A_none = calc_a_from_poles(self.J.poles)
        self.a = self.A_none[(GAMMA_C, 0)]          # = -9

    def test_a_matches_none(self):
        _, A = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='msd', n_msd=3)
        assert A == self.A_none

    def test_structure_keys(self):
        S, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='msd', n_msd=3)
        assert (np.inf, 0) in S
        assert coeff_at(S, GAMMA_C) != 0
        for nk in matsubara_freqs(3, T_TEST):
            assert coeff_at(S, nk) != 0
        # gamma_c pole + 3 Matsubara poles + 1 delta term
        assert len(S) == 5

    def test_gamma_c_coefficient_exact(self):
        # Full-coth residue at the J pole: -a cot(gc/2T), independent of n_msd.
        S, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='msd', n_msd=4)
        cot = 1.0 / np.tan(GAMMA_C / (2 * T_TEST))
        assert coeff_at(S, GAMMA_C).real == pytest.approx(-self.a * cot, rel=1e-12)

    def test_first_matsubara_coefficient_exact(self):
        S, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='msd', n_msd=4)
        nu1 = matsubara_freqs(1, T_TEST)[0]
        expected = 2 * T_TEST * ETA * GAMMA_C**2 * nu1 / (nu1**2 - GAMMA_C**2)
        assert coeff_at(S, nu1).real == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("n_msd", [0, 1, 2, 4])
    def test_s_delta_equals_matsubara_tail(self, n_msd):
        # The terminator delta carries the integrated weight of the dropped
        # Matsubara tail: s_delta = -4 T a sum_{k>n_msd} 1/(nu_k^2 - gc^2).
        S, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='msd', n_msd=n_msd)
        s_delta = complex(S[(np.inf, 0)]).real
        N = 2_000_000
        nu = 2 * np.pi * np.arange(n_msd + 1, N + 1) * T_TEST
        tail = -4 * T_TEST * self.a * np.sum(1.0 / (nu**2 - GAMMA_C**2))
        assert s_delta == pytest.approx(tail, abs=1e-5)

    def test_reconstruction_converges(self):
        Sx = S_exact_drude(ETA, GAMMA_C, T_TEST, T_GRID)
        errs = []
        for n_msd in [2, 5, 10, 50]:
            S, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='msd', n_msd=n_msd)
            errs.append(np.max(np.abs(reconstruct_S(S, T_GRID) - Sx)))
        for i in range(len(errs) - 1):
            assert errs[i + 1] < errs[i]
        assert errs[2] < 1e-7       # n_msd=10
        assert errs[3] < 1e-12      # n_msd=50


# ---------------------------------------------------------------------------
# calc_s_msd directly: synthetic single A-pole
# ---------------------------------------------------------------------------

class TestCalcSMsd:
    def test_single_pole_coefficients(self):
        # One A-pole at gamma=2.0, coefficient a=-1.0, temperature T=0.5.
        gamma_k = np.array([2.0], dtype=np.complex128)
        a_k = np.array([-1.0], dtype=np.complex128)
        T, n_ltc = 0.5, 3
        res = calc_s_msd(gamma_k, a_k, T, n_ltc)
        cot = 1.0 / np.tan(2.0 / (2 * T))
        assert coeff_at(res, 2.0).real == pytest.approx(1.0 * cot, rel=1e-12)
        for nk in matsubara_freqs(n_ltc, T):
            expected = T * nk * (-4 * (-1.0)) / (nk**2 - 2.0**2)
            assert coeff_at(res, nk).real == pytest.approx(expected, rel=1e-12)

    def test_degenerate_frequency_raises(self):
        # gamma equal to nu_1 = 2 pi T triggers the degeneracy guard.
        T = 0.5
        gamma_k = np.array([2 * np.pi * T], dtype=np.complex128)
        a_k = np.array([1.0], dtype=np.complex128)
        with pytest.raises(Exception):
            calc_s_msd(gamma_k, a_k, T, 2)


# ---------------------------------------------------------------------------
# psd (Pade spectral decomposition)
# ---------------------------------------------------------------------------

class TestPSD:
    def setup_method(self):
        self.J = Drude(eta=ETA, gamma_c=GAMMA_C)
        self.A_none = calc_a_from_poles(self.J.poles)

    def test_a_matches_none(self):
        _, A = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='psd',
                                    n_psd=2, type_psd='n-1/n')
        assert A == self.A_none

    @pytest.mark.parametrize("type_psd,bound", [('n-1/n', 1e-4), ('n/n', 1e-5)])
    def test_reconstruction_accuracy_and_convergence(self, type_psd, bound):
        Sx = S_exact_drude(ETA, GAMMA_C, T_TEST, T_GRID)
        errs = []
        for n_psd in [1, 2, 3, 5]:
            S, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='psd',
                                        n_psd=n_psd, type_psd=type_psd)
            errs.append(np.max(np.abs(reconstruct_S(S, T_GRID) - Sx)))
        for i in range(len(errs) - 1):
            assert errs[i + 1] < errs[i]
        assert errs[-1] < bound      # n_psd=5


# ---------------------------------------------------------------------------
# psd+fsd (Pade with Fano spectral decomposition correction)
# ---------------------------------------------------------------------------

class TestPSDFSD:
    def setup_method(self):
        self.J = Drude(eta=ETA, gamma_c=GAMMA_C)
        self.A_none = calc_a_from_poles(self.J.poles)

    def test_a_matches_none(self):
        _, A = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='psd+fsd',
                                    n_psd=2, type_psd='n-1/n',
                                    n_fsd_rec=1, chi_fsd=100.0)
        assert A == self.A_none

    def test_adds_poles_over_plain_psd(self):
        S_psd, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='psd',
                                        n_psd=1, type_psd='n-1/n')
        S_fsd, _ = calc_bath_corr_poles(self.J, T=T_TEST, type_ltc='psd+fsd',
                                        n_psd=1, type_psd='n-1/n',
                                        n_fsd_rec=1, chi_fsd=100.0)
        assert len(S_fsd) > len(S_psd)

    @pytest.mark.parametrize("n_psd", [1, 2, 3])
    def test_improves_low_temperature(self, n_psd):
        # At low T the plain Pade series is poor; FSD restores accuracy.
        T_low = 0.2
        Sx = S_exact_drude(ETA, GAMMA_C, T_low, T_GRID)
        S_psd, _ = calc_bath_corr_poles(self.J, T=T_low, type_ltc='psd',
                                        n_psd=n_psd, type_psd='n-1/n')
        S_fsd, _ = calc_bath_corr_poles(self.J, T=T_low, type_ltc='psd+fsd',
                                        n_psd=n_psd, type_psd='n-1/n',
                                        n_fsd_rec=1, chi_fsd=100.0)
        err_psd = np.max(np.abs(reconstruct_S(S_psd, T_GRID) - Sx))
        err_fsd = np.max(np.abs(reconstruct_S(S_fsd, T_GRID) - Sx))
        assert err_fsd < err_psd
        assert err_fsd < 0.05
