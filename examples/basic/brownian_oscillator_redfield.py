#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/
"""2-level Brownian-oscillator Redfield simulation (dimensionless units).

Same system as brownian_oscillator_heom.py.  The Redfield (Markovian)
approximation treats the bath as memoryless, so the bath oscillation visible
in the HEOM result does not appear here.  Compare both output files to assess
the role of non-Markovian bath memory.

Writes time, populations rho_00/rho_11, coherence Re/Im(rho_01), and
trace Tr(rho) to brownian_oscillator_redfield.dat.
"""

import time
from sys import stderr

import numpy as np
import tqdm

from pyheom import RedfieldSolver, noise_decomposition, Brown

# --- system parameters (all dimensionless) ---
lambda_0 = 0.1
omega_0  = 1.0
zeta     = 0.1
T        = 1.0

# --- Hamiltonian and coupling operator ---
omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)

c = 1.0
V = np.array([[c, 1.0], [1.0, -c]], dtype=np.complex128)

# --- bath correlation decomposition ---
J    = Brown(lambda_0, zeta, omega_0)
corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
corr.V = V

# --- solver setup ---
qme = RedfieldSolver(
    H, [corr],
    space='liouville', format='dense', engine='eigen',
    liouville_order='C', solver='lsrk4',
)

# --- initial state: excited state |0><0| ---
rho_0 = np.zeros((2, 2), dtype=np.complex128)
rho_0[0, 0] = 1.0

# --- print solver summary ---
print(f'Redfield  n_level={qme.n_level}', file=stderr)
print(f'          lambda_0={lambda_0}  omega_0={omega_0}  zeta={zeta}  T={T}  c={c}', file=stderr)

# --- time evolution ---
callback_dt = 2.5e-2
t_end       = 60.0
t_list      = np.arange(0.0, t_end, callback_dt)

_fmt = '{:12.6f}  {:14.10f}  {:14.10f}  {:+14.10f}  {:+14.10f}  {:14.10f}'

with open('brownian_oscillator_redfield.dat', 'w') as out, tqdm.tqdm(total=t_end) as bar:
    print(f'# lambda_0={lambda_0}  omega_0={omega_0}  zeta={zeta}  T={T}  c={c}', file=out)
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
    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)
    print(f'elapsed: {time.time() - t0:.1f} s', file=stderr)
