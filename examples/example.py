import numpy as np
import scipy as sp
import scipy.sparse

from sys import stdout, stderr
import time

import pyheom
pyheom.units['energy'] = pyheom.UNIT.dimensionless
pyheom.units['time']   = pyheom.UNIT.dimensionless
import tqdm

lambda_0  = 0.01 # reorganization energy (dimensionless)
omega_0   = 1    # vibrational frequency (dimensionless)
zeta      = 0.5  # damping constant      (dimensionless)
T         = 1    # temperature           (dimensionless)
max_tier  = 5

J = pyheom.Brownian(lambda_0, zeta, omega_0)

corr_dict = pyheom.noise_decomposition(
    J,
    T = T,
    type_LTC = 'PSD',
    n_PSD = 1,
    type_PSD = 'N-1/N')

n_state = 2

omega_1 = np.sqrt(omega_0**2 - zeta**2*0.25)
H = np.array([[omega_1, 0],
              [0, 0]])

V = np.array([[0, 1],
              [1, 0]])

noises = [
    dict(V=V, C=corr_dict)
]

h = pyheom.HEOM(H,
                noises,
                max_tier=max_tier,
                # matrix_type='sparse',
                # hierarchy_connection='hierarchical-Liouville',
                matrix_type='dense',
                hierarchy_connection='loop',
                # gpu_device=0
)
    
dt__unit = 0.25e-2
          
rho_0 = np.zeros((n_state,n_state))
rho_0[0,0] = 1
h.set_rho(rho_0)
            
callback_interval = 10*1
count             = 10*1000
t_total           = dt__unit*count

pbar = tqdm.tqdm(total=count)
with open('pop.dat', 'w') as out:
    print('# density matrix dynamics',   file=out)
    print('# time diabatic populations', file=out)
    def callback(t, rho):
        pbar.update(callback_interval)
        print(t, rho[0,0].real, rho[1,1].real, file=out)
        out.flush()
    begin = time.time()
    h.time_evolution(dt__unit, count, callback, callback_interval)
    end   = time.time()
pbar.close()
print('elapsed:', end - begin, file=stderr)
h = None
