#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for license.
# ------------------------------------------------------------------------*/
"""2-level Brownian-oscillator HEOM simulation (dimensionless units).

Writes time, populations rho_00/rho_11, coherence Re/Im(rho_01), and
trace Tr(rho) to brownian_oscillator_heom.dat.
"""

import time
from sys import stderr

import numpy as np
import tqdm

from pyheom import HEOMSolver, noise_decomposition, Brown

# --- system parameters (all dimensionless) ---
lambda_0 = 0.1
omega_0  = 1.0
zeta     = 0.5
T        = 1.0
J        = 0.1
truncation_depth  = 10

# --- Hamiltonian ---
omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
H = np.array([[omega_1, J], [J, 0.0]], dtype=np.complex128)

# --- coupling operator ---
V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

# --- bath correlation decomposition ---
J_sd = Brown(lambda_0, zeta, omega_0)
corr = noise_decomposition(J_sd, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
corr.V = V

# --- solver setup ---
qme = HEOMSolver(
    H, [corr],
    space='Liouville', format='dense', engine='Eigen',
    liouville_order='C', solver='rkdp',
    truncation_depth=truncation_depth, n_inner_threads=4, n_outer_threads=1,
)

# --- initial state: excited state |0><0| ---
rho_0 = np.zeros((2, 2), dtype=np.complex128)
rho_0[0, 0] = 1.0

# --- print solver summary ---
n_hierarchy = qme.init(rho_0, dt=1e-2).rho_hierarchy.shape[0]
print(f'HEOM  n_level={qme.n_level}  truncation_depth={truncation_depth}  n_hierarchy={n_hierarchy}', file=stderr)
print(f'      lambda_0={lambda_0}  omega_0={omega_0}  zeta={zeta}  T={T}  J={J}', file=stderr)

# --- time evolution ---
callback_dt = 0.1
t_end       = 80.0
t_list      = np.arange(0.0, t_end, callback_dt)

_fmt = '{:12.6f}  {:14.10f}  {:14.10f}  {:+14.10f}  {:+14.10f}  {:14.10f}'

with open('brownian_oscillator_heom.dat', 'w') as out, tqdm.tqdm(total=t_end) as bar:
    print(f'# lambda_0={lambda_0}  omega_0={omega_0}  zeta={zeta}  T={T}'
          f'  J={J}  truncation_depth={truncation_depth}  n_hierarchy={n_hierarchy}', file=out)
    print('# {:>10s}  {:>14s}  {:>14s}  {:>14s}  {:>14s}  {:>14s}'.format(
        't', 'rho_00', 'rho_11', 'Re(rho_01)', 'Im(rho_01)', 'Tr(rho)'), file=out)
    def callback(t):
        bar.update(callback_dt)
        rho = qme.rho
        coh = rho[0, 1]
        tr  = rho[0, 0].real + rho[1, 1].real
        print(_fmt.format(t, rho[0,0].real, rho[1,1].real, coh.real, coh.imag, tr), file=out)
        out.flush()
    t0 = time.time()
    qme.solve(rho_0, t_list, callback=callback, dt=1e-2, atol=1e-8, rtol=1e-6)
    print(f'elapsed: {time.time() - t0:.1f} s', file=stderr)
