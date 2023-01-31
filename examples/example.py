#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np
import scipy as sp
import scipy.sparse

from sys import stdout, stderr
import time

import pyheom
from pyheom import heom_solver, redfield_solver, noise_decomposition, brown, unit
pyheom.units['energy'] = unit.dimensionless
pyheom.units['time']   = unit.dimensionless
import tqdm

dtype           = np.complex128
space           = 'liouville'
format          = 'dense'
engine          = 'eigen'
solver          = 'lsrk4'
order_liouville = 'row_major'

lambda_0  = 0.01 # coupling constant     (dimensionless)
omega_0   = 1    # vibrational frequency (dimensionless)
zeta      = 0.5  # damping constant      (dimensionless)
T         = 1    # temperature           (dimensionless)
depth     = 5

callback_interval = 2.5e-2
count             = 25
t_list            = np.arange(0, count, callback_interval)
solver_params    = dict(
    dt = 0.25e-2,
    # atol=1e-6, rtol=1e-3
)
# 

J = brown(lambda_0, zeta, omega_0)

corr_dict = noise_decomposition(
    J,
    T = T,
    type_ltc = 'psd',
    n_psd = 1,
    type_psd = 'n-1/n'
)

n_level = 2

omega_1 = np.sqrt(omega_0**2 - zeta**2*0.25)
H = np.array([[omega_1, 0],
              [0, 0]],
             dtype=dtype)

V = np.array([[0, 1],
              [1, 0]],
             dtype=dtype)

qme = heom_solver(H, [dict(V=V, **corr_dict)],
                  space=space, format=format, engine=engine,
                  order_liouville=order_liouville,
                  solver=solver,
                  engine_args=dict(),
                  depth = depth,
                  n_inner_threads = 4,
                  n_outer_threads = 1)

n_storage = qme.storage_size()


rho = np.zeros((n_storage,n_level,n_level), dtype=dtype)
rho_0 = rho[0,:,:]
rho_0[0,0] = 1

with open('pop.dat', 'w') as out, \
     tqdm.tqdm(total=count) as bar:
    print('# density matrix dynamics',   file=out)
    print('# time diabatic populations', file=out)
    def callback(t):
        bar.update(callback_interval)
        print(t, rho_0[0,0].real, rho_0[1,1].real, file=out)
        out.flush()
    begin = time.time()
    qme.solve(rho, t_list, callback=callback, **solver_params)
    end   = time.time()
print('elapsed:', end - begin, file=stderr)
del qme
