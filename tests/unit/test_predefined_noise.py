"""Tests for pyheom.predefined_noise: spectral densities at known frequencies.

Numerical-contract style: pole decompositions and analytic J(omega) values are
pinned so that future renames only require import-line changes.
"""

import numpy as np
import pytest

from pyheom.predefined_noise import (
    SpectralDensity, Drude, Brown, OverdampedBrown, BrownDrude,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: evaluate J(omega) from the pole representation
# ---------------------------------------------------------------------------

def spectrum_from_poles(poles, omega):
    """J(omega) reconstructed from [(gamma, coef, m, n), ...]."""
    total = sum(coef * omega**(2*n+1) / (gamma**2 + omega**2)**m
                for gamma, coef, m, n in poles)
    return float(np.real(total))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TestSpectralDensityABC:
    def test_all_classes_inherit_abc(self):
        for cls in (Drude, Brown, OverdampedBrown, BrownDrude):
            assert issubclass(cls, SpectralDensity)

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SpectralDensity()


# ---------------------------------------------------------------------------
# Drude
# ---------------------------------------------------------------------------

class TestDrude:
    # J(omega) = eta * gamma_c^2 * omega / (omega^2 + gamma_c^2)
    # Drude(eta=2, gamma_c=3): J(omega) = 18*omega/(omega^2+9)

    def setup_method(self):
        self.J = Drude(eta=2.0, gamma_c=3.0)

    def test_name(self):
        assert self.J.name == 'drude'

    @pytest.mark.parametrize("omega,expected", [
        (1.0, 18 / 10),    # 18*1/(1+9) = 1.8
        (3.0, 54 / 18),    # 18*3/(9+9) = 3.0
        (6.0, 108 / 45),   # 18*6/(36+9) = 2.4
    ])
    def test_spectrum(self, omega, expected):
        assert self.J.spectrum(omega) == pytest.approx(expected, rel=1e-12)

    def test_poles_structure(self):
        assert len(self.J.poles) == 1
        gamma, coef, m, n = self.J.poles[0]
        assert float(gamma) == pytest.approx(3.0)
        assert float(coef)  == pytest.approx(2.0 * 9.0)
        assert m == 1 and n == 0

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, 5.0])
    def test_poles_consistency(self, omega):
        assert spectrum_from_poles(self.J.poles, omega) == pytest.approx(
            self.J.spectrum(omega), rel=1e-12)


# ---------------------------------------------------------------------------
# Brown -- underdamped  (omega_0 > gamma_c/2, complex poles)
# ---------------------------------------------------------------------------

class TestBrownUnderdamped:
    # lambda_0=1, gamma_c=0.5, omega_0=2
    # omega_1 = sqrt(4 - 0.0625) = sqrt(3.9375) (real)
    # J(omega) = 4*omega / ((4-omega^2)^2 + 0.25*omega^2)

    def setup_method(self):
        self.J = Brown(lambda_0=1.0, gamma_c=0.5, omega_0=2.0)

    def test_name(self):
        assert self.J.name == 'brown'

    @pytest.mark.parametrize("omega,expected", [
        (1.0, 4 / 9.25),   # 4/(9+0.25)
        (2.0, 8.0),         # at resonance: 8/((0)^2 + 0.25*4)
        (4.0, 16 / 148.0),  # 16/(144+4)
    ])
    def test_spectrum(self, omega, expected):
        assert self.J.spectrum(omega) == pytest.approx(expected, rel=1e-12)

    def test_poles_count(self):
        assert len(self.J.poles) == 2

    def test_poles_are_complex(self):
        # underdamped -> complex-conjugate pair
        gamma_0 = self.J.poles[0][0]
        gamma_1 = self.J.poles[1][0]
        assert isinstance(gamma_0, complex)
        assert gamma_1 == pytest.approx(gamma_0.conjugate())

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, 4.0])
    def test_poles_consistency(self, omega):
        assert spectrum_from_poles(self.J.poles, omega) == pytest.approx(
            self.J.spectrum(omega), rel=1e-10)


# ---------------------------------------------------------------------------
# Brown -- overdamped  (gamma_c/2 > omega_0, real poles)
# ---------------------------------------------------------------------------

class TestBrownOverdamped:
    # lambda_0=1, gamma_c=4, omega_0=1
    # J(omega) = 8*omega / ((1-omega^2)^2 + 16*omega^2)

    def setup_method(self):
        self.J = Brown(lambda_0=1.0, gamma_c=4.0, omega_0=1.0)

    @pytest.mark.parametrize("omega,expected", [
        (1.0, 8 / 16),          # 8/16 = 0.5
        (0.5, 4 / 4.5625),      # 8*0.5/((0.75)^2 + 4)
    ])
    def test_spectrum(self, omega, expected):
        assert self.J.spectrum(omega) == pytest.approx(expected, rel=1e-12)

    def test_poles_are_real(self):
        # overdamped -> two real poles (stored as float)
        assert len(self.J.poles) == 2
        for gamma, coef, m, n in self.J.poles:
            assert isinstance(gamma, float)
            assert gamma > 0

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0])
    def test_poles_consistency(self, omega):
        assert spectrum_from_poles(self.J.poles, omega) == pytest.approx(
            self.J.spectrum(omega), rel=1e-12)


# ---------------------------------------------------------------------------
# Brown -- critically damped  (omega_0 == gamma_c/2, second-order pole)
# ---------------------------------------------------------------------------

class TestBrownCritical:
    # lambda_0=1, gamma_c=2, omega_0=1 -> omega_1 = 0
    # J(omega) = 4*omega / ((1-omega^2)^2 + 4*omega^2)

    def setup_method(self):
        self.J = Brown(lambda_0=1.0, gamma_c=2.0, omega_0=1.0)

    @pytest.mark.parametrize("omega,expected", [
        (1.0, 1.0),      # 4/(0+4)
        (0.5, 1.28),     # 2/((0.75)^2 + 1) = 2/1.5625
    ])
    def test_spectrum(self, omega, expected):
        assert self.J.spectrum(omega) == pytest.approx(expected, rel=1e-12)

    def test_poles_single_second_order(self):
        assert len(self.J.poles) == 1
        gamma, coef, m, n = self.J.poles[0]
        assert m == 2  # second-order pole
        assert float(gamma) == pytest.approx(1.0)   # gamma_c/2
        assert float(coef)  == pytest.approx(4.0)   # 2*lambda_0*gamma_c*omega_0^2

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0])
    def test_poles_consistency(self, omega):
        assert spectrum_from_poles(self.J.poles, omega) == pytest.approx(
            self.J.spectrum(omega), rel=1e-12)


# ---------------------------------------------------------------------------
# OverdampedBrown
# ---------------------------------------------------------------------------

class TestOverdampedBrown:
    # lambda_0=1, gamma_c=4, omega_0=2
    # gamma_c_eff = omega_0^2/gamma_c = 4/4 = 1
    # eta_eff = 2*lambda_0/gamma_c_eff = 2
    # -> Drude(eta=2, gamma_c=1): J = 2*omega/(omega^2+1)

    def setup_method(self):
        self.J = OverdampedBrown(lambda_0=1.0, gamma_c=4.0, omega_0=2.0)

    def test_is_drude_subclass(self):
        assert isinstance(self.J, Drude)

    def test_name(self):
        assert self.J.name == 'overdamped_brown'

    def test_effective_params(self):
        assert self.J.gamma_c == pytest.approx(1.0)
        assert self.J.eta     == pytest.approx(2.0)

    @pytest.mark.parametrize("omega,expected", [
        (1.0, 1.0),   # 2/2
        (2.0, 0.8),   # 4/5
    ])
    def test_spectrum(self, omega, expected):
        assert self.J.spectrum(omega) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0])
    def test_poles_consistency(self, omega):
        assert spectrum_from_poles(self.J.poles, omega) == pytest.approx(
            self.J.spectrum(omega), rel=1e-12)


# ---------------------------------------------------------------------------
# BrownDrude
# ---------------------------------------------------------------------------

class TestBrownDrude:
    # lambda_0=1, zeta=0.5, gamma_c=2, omega_0=1
    # delta_omega2 = omega^2/(omega^2+4)
    # delta_gamma2 = 2*omega/(omega^2+4)
    # J(omega) = 2*delta_gamma2 / ((1-omega^2+delta_omega2)^2 + delta_gamma2^2)

    def setup_method(self):
        self.J = BrownDrude(lambda_0=1.0, zeta=0.5, gamma_c=2.0, omega_0=1.0)

    def test_name(self):
        assert self.J.name == 'brown_drude'

    @pytest.mark.parametrize("omega,expected", [
        # omega=1: delta_omega2=0.2, delta_gamma2=0.4
        # J = 2*0.4/((0.2)^2 + 0.4^2) = 0.8/0.2 = 4.0
        (1.0, 4.0),
        # omega=2: delta_omega2=0.5, delta_gamma2=0.5
        # J = 2*0.5/((-2.5)^2 + 0.5^2) = 1.0/6.5
        (2.0, 1.0 / 6.5),
    ])
    def test_spectrum(self, omega, expected):
        assert self.J.spectrum(omega) == pytest.approx(expected, rel=1e-12)

    def test_poles_count(self):
        assert len(self.J.poles) == 3  # cubic characteristic equation -> 3 poles

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, 3.0])
    def test_poles_consistency(self, omega):
        assert spectrum_from_poles(self.J.poles, omega) == pytest.approx(
            self.J.spectrum(omega), rel=1e-10)
