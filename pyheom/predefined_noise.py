#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np
import cmath as cm

class drude:
    def __init__(self, eta, gamma_c):
        self.eta     = float(eta)
        self.gamma_c = float(gamma_c)
        self.name = 'drude'
        self.poles = self.get_poles()

    def get_poles(self):
        eta     = self.eta
        gamma_c = self.gamma_c
        return [[gamma_c, eta*gamma_c**2, 1, 0]]

    def spectrum(self, omega):
        eta     = self.eta
        gamma_c = self.gamma_c
        return eta*gamma_c**2*omega/(omega**2+gamma_c**2)


class brown:
    def __init__(self, lambda_0, gamma_c, omega_0):
        self.lambda_0 = float(lambda_0)
        self.gamma_c  = float(gamma_c)
        self.omega_0  = float(omega_0)
        self.name = 'brown'
        self.poles = self.get_poles()

    def get_poles(self):
        lambda_0 = self.lambda_0
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0
        
        omega_1 = cm.sqrt(omega_0**2 - gamma_c**2*0.25)

        if np.abs(omega_1) < (np.finfo(float).eps):
            # critical damped
            coef = 2*lambda_0*gamma_c*omega_0**2
            return [[gamma_c*0.5, coef, 2, 0]]
        
        gamma_p = gamma_c*0.5 + 1.0j*omega_1
        coef_p  = +1.0j*lambda_0*omega_0**2/omega_1
        gamma_m = gamma_c*0.5 - 1.0j*omega_1
        coef_m  = -1.0j*lambda_0*omega_0**2/omega_1

        if gamma_p.imag == 0.0:
            # overdamped
            return [[gamma_p.real, coef_p.real, 1, 0],
                    [gamma_m.real, coef_m.real, 1, 0]]
        else:
            # underdamped
            return [[gamma_p, coef_p, 1, 0],
                    [gamma_m, coef_m, 1, 0]]

    def spectrum(self, omega):
        lambda_0 = self.lambda_0
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0
        
        return 2*lambda_0*gamma_c*omega_0**2*omega \
            /((omega_0**2 - omega**2)**2 + gamma_c**2*omega**2)


class overdamped_brown(drude):
    def __init__(self, lambda_0, gamma_c, omega_0):
        self.gamma_c = float(omega_0**2/gamma_c)
        self.eta     = float(2*lambda_0/self.gamma_c)
        self.name    = 'overdamped_brown'
        self.poles   = self.get_poles()


class brown_drude:
    def __init__(self, lambda_0, zeta, gamma_c, omega_0):
        self.lambda_0 = float(lambda_0)
        self.zeta     = float(zeta)
        self.gamma_c  = float(gamma_c)
        self.omega_0  = float(omega_0)
    
    def get_name(self):
        return 'brown_drude'

    def calc_r_k(self):
        lambda_0 = self.lambda_0
        zeta     = self.zeta
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0

        c3 = 1.0
        c2 = -2*omega_0**2 + gamma_c**2 - 2*zeta*gamma_c
        c1 = omega_0**4 + (-2*gamma_c**2 + 2*zeta*gamma_c)*omega_0**2 + zeta**2*gamma_c**2
        c0 = gamma_c**2*omega_0**4

        omega2_k = np.roots([c3, c2, c1, c0])
        gamma_k = np.array([cm.sqrt(omega2)/1.0j if cm.sqrt(omega2).imag >= 0.0 else -cm.sqrt(omega2)/1.0j for omega2 in omega2_k])
        # if np.abs(omega_1) < (np.finfo(float).eps):
        #     raise Exception('Calculation for critical damping case is unsupported feature.')

        Gamma = (gamma_k[0]**2-gamma_k[1]**2)*(gamma_k[1]**2-gamma_k[2]**2)*(gamma_k[2]**2-gamma_k[0]**2)

        coef_t = -2*lambda_0*omega_0**2*zeta*gamma_c**2/Gamma
        coef_0 = coef_t*(gamma_k[1]**2 - gamma_k[2]**2)
        coef_1 = coef_t*(gamma_k[2]**2 - gamma_k[0]**2)
        coef_2 = coef_t*(gamma_k[0]**2 - gamma_k[1]**2)
        
        r_k    = np.array([coef_0, coef_1, coef_2])
        return [[gamma_k[0], coef_0, 1, 0],
                [gamma_k[1], coef_1, 1, 0],
                [gamma_k[2], coef_2, 1, 0]]

    def spectrum(self, omega):
        lambda_0 = self.lambda_0
        zeta     = self.zeta
        gamma_c  = self.gamma_c
        omega_0  = self.omega_0

        deltaOmega2 = zeta*omega**2*gamma_c/(omega**2 + gamma_c**2)
        deltaGamma2 = zeta*omega*gamma_c**2/(omega**2 + gamma_c**2)
            
        return 2.0*lambda_0*omega_0**2*deltaGamma2 \
            /((self.omega_0**2 - omega**2 + deltaOmega2)**2 + deltaGamma2**2)    
