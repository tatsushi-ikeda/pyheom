#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/
"""2-level Brownian-oscillator HEOM simulation (dimensionless units).

V = sigma_x + c*sigma_z mixes a coupling mode (sigma_x) and a tuning mode
(sigma_z).  The sigma_z component breaks the Z2 symmetry that kept coherences
zero for pure sigma_x coupling, so rho_01 is now non-zero.  The sigma_z term
shifts both energy levels equally, leaving the gap unchanged; no reorganization
energy correction to H is needed.

With a lightly damped (underdamped) bath (zeta << omega_0), the bath
correlation function oscillates at frequency ~omega_0 before decaying.  This
oscillation is visible in both the coherences and populations.  Compare with
brownian_oscillator_redfield.py, where the Markovian approximation suppresses
the bath memory and the oscillation is absent.

Writes time, populations rho_00/rho_11, coherence Re/Im(rho_01), and
trace Tr(rho) to brownian_oscillator_heom.dat.
"""

import time
from sys import stderr

import numpy as np
import tqdm

from pyheom import HEOMSolver, noise_decomposition, Brown

# --- system parameters (all dimensionless) ---
lambda_0 = 0.1    # bath reorganization energy (stronger than minimal to see oscillation)
omega_0  = 1.0    # bath oscillator frequency; oscillation period ~ 2*pi/omega_0 ~ 6.3
zeta     = 0.1    # lightly damped: bath memory persists ~1/0.05 ~ 20 time units
T        = 1.0    # temperature
n_tiers  = 15     # hierarchy depth (more needed for stronger coupling and light damping)

# --- Hamiltonian: diagonal, energy gap = omega_1 (renormalized bath frequency) ---
omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)

# --- coupling: sigma_x (coupling mode) + sigma_z (tuning mode) ---
c = 1.0   # tuning mode weight relative to coupling mode
V = np.array([[c, 1.0], [1.0, -c]], dtype=np.complex128)

# --- bath correlation decomposition ---
J    = Brown(lambda_0, zeta, omega_0)
corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
corr.V = V

# --- solver setup ---
qme = HEOMSolver(
    H, [corr],
    space='liouville', format='dense', engine='eigen',
    liouville_order='C', solver='lsrk4',
    n_tiers=n_tiers, n_inner_threads=4, n_outer_threads=1,
)

# --- initial state: excited state |0><0| ---
rho_0 = np.zeros((2, 2), dtype=np.complex128)
rho_0[0, 0] = 1.0

# --- print solver summary ---
n_hrchy = qme.init(rho_0, dt=0.25e-2).rho_hierarchy.shape[0]
print(f'HEOM  n_level={qme.n_level}  n_tiers={n_tiers}  n_hrchy={n_hrchy}', file=stderr)
print(f'      lambda_0={lambda_0}  omega_0={omega_0}  zeta={zeta}  T={T}  c={c}', file=stderr)

# --- time evolution ---
callback_dt = 2.5e-2
t_end       = 60.0
t_list      = np.arange(0.0, t_end, callback_dt)

_fmt = '{:12.6f}  {:14.10f}  {:14.10f}  {:+14.10f}  {:+14.10f}  {:14.10f}'

with open('brownian_oscillator_heom.dat', 'w') as out, tqdm.tqdm(total=t_end) as bar:
    print(f'# lambda_0={lambda_0}  omega_0={omega_0}  zeta={zeta}  T={T}'
          f'  c={c}  n_tiers={n_tiers}  n_hrchy={n_hrchy}', file=out)
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
