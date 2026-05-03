#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/
"""2-level Brownian-oscillator HEOM simulation (dimensionless units).

Writes population dynamics to pop.dat.
"""

import time
from sys import stderr

import numpy as np
import tqdm

from pyheom import HEOMSolver, noise_decomposition, Brown

# --- system parameters (all dimensionless) ---
lambda_0 = 0.01   # coupling constant
omega_0  = 1.0    # vibrational frequency
zeta     = 0.5    # damping constant
T        = 1.0    # temperature
n_tiers  = 5

# --- Hamiltonian and coupling operator ---
omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)   # renormalised frequency
H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
V = np.array([[0.0, 1.0], [1.0, 0.0]],    dtype=np.complex128)

# --- bath correlation decomposition ---
J    = Brown(lambda_0, zeta, omega_0)
corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
corr.V = V

# --- solver setup ---
# Engine, space, and format can be selected automatically with:
#   qme = HEOMSolver.auto(H, [corr], n_tiers=n_tiers)
qme = HEOMSolver(
    H, [corr],
    space='liouville', format='dense', engine='eigen',
    liouville_order='C', solver='lsrk4',
    n_tiers=n_tiers, n_inner_threads=4, n_outer_threads=1,
)

# --- initial state: excited state |0><0| ---
rho_0 = np.zeros((2, 2), dtype=np.complex128)
rho_0[0, 0] = 1.0

# --- time evolution ---
callback_dt = 2.5e-2
t_end       = 25.0
t_list      = np.arange(0.0, t_end, callback_dt)

with open('pop.dat', 'w') as out, tqdm.tqdm(total=t_end) as bar:
    print('# time  rho_00  rho_11', file=out)
    def callback(t):
        bar.update(callback_dt)
        print(t, qme.rho[0, 0].real, qme.rho[1, 1].real, file=out)
        out.flush()
    t0 = time.time()
    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)
    print(f'elapsed: {time.time() - t0:.1f} s', file=stderr)
