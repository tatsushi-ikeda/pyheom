#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

import cmath as cm
from abc import ABC, abstractmethod

import numpy as np


class SpectralDensity(ABC):
    """Abstract base for spectral density models J(omega)."""

    name: str = ''

    @abstractmethod
    def spectrum(self, omega):
        """Return J(omega)."""

    @abstractmethod
    def get_poles(self):
        """Return a list of [gamma, coef, m, n] entries describing the poles."""


class Drude(SpectralDensity):
    """Drude (Ohmic with Lorentz cutoff) spectral density: J(omega) = eta gamma_c^2 omega / (omega^2 + gamma_c^2)."""
    name = 'drude'

    def __init__(self, eta, gamma_c):
        self.eta     = float(eta)
        self.gamma_c = float(gamma_c)
        self.poles   = self.get_poles()

    def get_poles(self):
        eta     = self.eta
        gamma_c = self.gamma_c
        return [[gamma_c, eta*gamma_c**2, 1, 0]]

    def spectrum(self, omega):
        eta     = self.eta
        gamma_c = self.gamma_c
        return eta*gamma_c**2*omega/(omega**2 + gamma_c**2)


class Brown(SpectralDensity):
    """Brownian-oscillator spectral density; overdamped/underdamped/critical depending on gamma_c vs 2omega_0."""
    name = 'brown'

    def __init__(self, lambda_0, gamma_c, omega_0):
        self.lambda_0 = float(lambda_0)
        self.gamma_c  = float(gamma_c)
        self.omega_0  = float(omega_0)
        self.poles    = self.get_poles()

    def get_poles(self):
        lambda_0 = self.lambda_0
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0

        omega_1 = cm.sqrt(omega_0**2 - gamma_c**2*0.25)

        if np.abs(omega_1) < np.finfo(float).eps:
            # critically damped
            coef = 2*lambda_0*gamma_c*omega_0**2
            return [[gamma_c*0.5, coef, 2, 0]]

        gamma_p = gamma_c*0.5 + 1.0j*omega_1
        coef_p  = +1.0j*lambda_0*omega_0**2/omega_1
        gamma_m = gamma_c*0.5 - 1.0j*omega_1
        coef_m  = -1.0j*lambda_0*omega_0**2/omega_1

        if gamma_p.imag == 0.0:
            # overdamped (omega_1 purely imaginary)
            return [[gamma_p.real, coef_p.real, 1, 0],
                    [gamma_m.real, coef_m.real, 1, 0]]
        # underdamped
        return [[gamma_p, coef_p, 1, 0],
                [gamma_m, coef_m, 1, 0]]

    def spectrum(self, omega):
        lambda_0 = self.lambda_0
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0
        return 2*lambda_0*gamma_c*omega_0**2*omega \
            / ((omega_0**2 - omega**2)**2 + gamma_c**2*omega**2)


class OverdampedBrown(Drude):
    """Brown spectral density in the overdamped limit, mapped to a Drude form.

    Effective parameters: gamma_c_eff = omega_0^2 / gamma_c,
    eta_eff = 2*lambda_0 / gamma_c_eff.
    """
    name = 'overdamped_brown'

    def __init__(self, lambda_0, gamma_c, omega_0):
        gamma_c_eff = float(omega_0**2/gamma_c)
        eta_eff     = float(2*lambda_0/gamma_c_eff)
        super().__init__(eta_eff, gamma_c_eff)


class BrownDrude(SpectralDensity):
    """Brown spectral density with frequency-dependent Drude friction kernel."""
    name = 'brown_drude'

    def __init__(self, lambda_0, zeta, gamma_c, omega_0):
        self.lambda_0 = float(lambda_0)
        self.zeta     = float(zeta)
        self.gamma_c  = float(gamma_c)
        self.omega_0  = float(omega_0)
        self.poles    = self.get_poles()

    def get_poles(self):
        lambda_0 = self.lambda_0
        zeta     = self.zeta
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0

        c3 = 1.0
        c2 = -2*omega_0**2 + gamma_c**2 - 2*zeta*gamma_c
        c1 = omega_0**4 + (-2*gamma_c**2 + 2*zeta*gamma_c)*omega_0**2 + zeta**2*gamma_c**2
        c0 = gamma_c**2*omega_0**4

        omega2_k = np.roots([c3, c2, c1, c0])
        gamma_k = np.array([
            cm.sqrt(o2)/1.0j if cm.sqrt(o2).imag >= 0.0 else -cm.sqrt(o2)/1.0j
            for o2 in omega2_k
        ])

        Gamma = ((gamma_k[0]**2 - gamma_k[1]**2)
                 *(gamma_k[1]**2 - gamma_k[2]**2)
                 *(gamma_k[2]**2 - gamma_k[0]**2))

        coef_t = -2*lambda_0*omega_0**2*zeta*gamma_c**2/Gamma
        coef_0 = coef_t*(gamma_k[1]**2 - gamma_k[2]**2)
        coef_1 = coef_t*(gamma_k[2]**2 - gamma_k[0]**2)
        coef_2 = coef_t*(gamma_k[0]**2 - gamma_k[1]**2)

        return [[gamma_k[0], coef_0, 1, 0],
                [gamma_k[1], coef_1, 1, 0],
                [gamma_k[2], coef_2, 1, 0]]

    def spectrum(self, omega):
        lambda_0 = self.lambda_0
        zeta     = self.zeta
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0

        delta_omega2 = zeta*omega**2*gamma_c   /(omega**2 + gamma_c**2)
        delta_gamma2 = zeta*omega   *gamma_c**2/(omega**2 + gamma_c**2)

        return 2.0*lambda_0*omega_0**2*delta_gamma2 \
            / ((omega_0**2 - omega**2 + delta_omega2)**2 + delta_gamma2**2)
