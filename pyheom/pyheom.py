# 
# LibHEOM: Copyright (c) Tatsushi Ikeda
# This library is distributed under BSD 3-Clause License.
# See LINCENSE.txt for licence.
# ------------------------------------------------------------------------

import enum
import sys
import numpy as np
import scipy as sp
import scipy.sparse
import importlib

pylibheom = importlib.import_module("pylibheom")
from pyheom.noise_decomposition import *

unit = enum.Enum('unit',
 '''dimensionless
    femtosecond
    picosecond
    wavenumber
    electronvolt''')

hbar__J_s = 1.05457180013e-34
UNIT_ENERGY_VALUE__J = {
    unit.wavenumber:   1.98644582441459e-23, # (299792458*100*6.62607004081e-34)
    unit.electronvolt: 1.602176620898e-19,
};
UNIT_TIME_VALUE__S = {
    unit.femtosecond: 1.0e-15,
    unit.picosecond:  1.0e-12,
}

units = {'energy':unit.dimensionless,
         'time':  unit.dimensionless}

def calc_unit():
    if (units['energy'] == unit.dimensionless or units['time'] == unit.dimensionless):
        if (units['energy'] == unit.dimensionless and units['time'] == unit.dimensionless):
            result = 1.0
        else:
            print('[Error] Unit mismatch error: Both unit_energy and unit_time should be dimensionless.', file=sys.stderr)
            sys.exit(1)
    else:
        result = (UNIT_ENERGY_VALUE__J[units['energy']]
                *UNIT_TIME_VALUE__S[units['time']]
                /hbar__J_s)
    return result


def get_coo_matrix(matrix):
    impl_class_name = "coo_matrix"
    if   matrix.dtype == np.complex64:
        ipml_class_name += "_c"
    elif matrix.dtype == np.complex128:
        impl_class_name += "_z"
    else:
        print('[Error] Unsupported matri type: {}.'.format(matrix.dtype),
              file=sys.stderr)
        sys.exit(1)
    coo = sp.sparse.coo_matrix(matrix)
    impl_class = getattr(pylibheom, impl_class_name)
    return impl_class(
        coo.shape[0],
        coo.shape[1],
        coo.nnz,
        coo.row,
        coo.col,
        coo.data)
    
class heom():
    def __init__(self,
                 H,
                 noises,
                 max_tier,
                 matrix_type='sparse',
                 hierarchy_connection='loop',
                 gpu_device=None,
                 callback=lambda lidx: None,
                 callback_interval=1024):
        impl_class_name = 'heom_z'

        if   matrix_type == 'dense':
            impl_class_name += 'd'
        elif matrix_type == 'sparse':
            impl_class_name += 's'
        else:
            print('[Error] Unknown internal matrix type: {}.'.format(
                matrix_type))
            sys.exit(1)
        
        impl_class_name += 'l'

        if   hierarchy_connection == 'loop':
            impl_class_name += 'l'
        elif hierarchy_connection == 'hierarchical-Liouville':
            impl_class_name += 'h'
        else:
            print('[Error] Unknown internal hierarchy connection: {}.'.format(
                hierarchy_connection))
            sys.exit(1)
        
        if (not gpu_device is None):
            if getattr(pylibheom, 'support_gpu_parallelization'):
                impl_class_name += '_gpu'
            else:
                print('[Error] gpu parallelization is not supported.')
                print('  specified gpu device: {}.'.format(gpu_device))
                sys.exit(1)

        self.impl = getattr(pylibheom, impl_class_name)()
        
        if (not gpu_device is None):
            self.impl.set_device_number(gpu_device)
        
        self.n_state = H.shape[0]
        self.impl.set_hamiltonian(get_coo_matrix(H.astype(np.complex128)))

        n_noise = len(noises)
        self.impl.allocate_noises(n_noise)

        for u in range(n_noise):
            gamma   = noises[u]["C"]["gamma"]
            phi_0   = noises[u]["C"]["phi_0"]
            sigma   = noises[u]["C"]["sigma"]
            s       = noises[u]["C"]["s"]
            a       = noises[u]["C"]["a"]
            S_delta = noises[u]["C"]["S_delta"]
            self.impl.set_noise(u,
                                get_coo_matrix(noises[u]["V"].astype(np.complex128)),
                                get_coo_matrix(gamma.astype(np.complex128)),
                                phi_0.astype(np.complex128),
                                sigma.astype(np.complex128),
                                get_coo_matrix(s.astype(np.complex128)),
                                complex(S_delta),
                                get_coo_matrix(a.astype(np.complex128)))


        self.impl.flatten_hierarchy_dimension()
        self.n_hierarchy \
            = self.impl.allocate_hierarchy_space(max_tier,
                                                 callback,
                                                 callback_interval)
        self.rho_h = np.zeros((self.n_state, self.n_state, self.n_hierarchy),
                              dtype=np.complex128, order='F')
        
        self.impl.init_aux_vars(callback,
                                callback_interval)
        
    def construct_commutator(self,
                             x, coef_l, coef_r,
                             callback=lambda lidx: None,
                             callback_interval=1024):
        x_coo = sp.sparse.coo_matrix(x)
        self.impl.construct_commutator(x_coo.shape[0],
                                       x_coo.shape[1],
                                       x_coo.nnz,
                                       x_coo.row,
                                       x_coo.col,
                                       x_coo.data.astype(np.complex128),
                                       coef_l,
                                       coef_r,
                                       callback,
                                       callback_interval)

    def apply_commutator(self):
        self.impl.apply_commutator(self.rho_h.ravel())

    def set_rho(self, rho):
        self.rho_h[:,:,0] = rho[:,:]

    def get_rho(self):
        return np.copy(self.rho_h[:,:,0])

    def set_rho_h(self, rho_h):
        self.rho_h[:,:,:] = rho_h[:,:,:]

    def get_rho_h(self):
        return np.copy(self.rho_h[:,:,:])

    def time_evolution(self, dt__unit, count,
                       callback=lambda t, rho: None,
                       callback_interval=1):
        self.impl.time_evolution(self.rho_h.ravel(order='F'),
                                 dt__unit, dt__unit*calc_unit(),
                                 callback_interval, count//callback_interval,
                                 lambda t: callback(t, self.rho_h[:,:,0]))


class redfield():
    def __init__(self,
                 H,
                 noises,
                 matrix_type='sparse',
                 operator_space='Liouville',
                 gpu_device=None,
                 callback=lambda lidx: None,
                 callback_interval=1024):
        impl_class_name = 'redfield_z'

        if   matrix_type == 'dense':
            impl_class_name += 'd'
        elif matrix_type == 'sparse':
            impl_class_name += 's'
        else:
            print('[Error] Unknown internal matrix type: {}.'.format(
                matrix_type))
            sys.exit(1)

        if   operator_space == 'Hilbert':
            impl_class_name += 'h'
        elif operator_space == 'Liouville':
            impl_class_name += 'l'
        else:
            print('[Error] Unknown internal operator space: {}.'.format(
                operator_space))
            sys.exit(1)
        
        if (not gpu_device is None):
            if support_gpu_parallelization:
                impl_class_name += '_gpu'
            else:
                print('[Error] gpu parallelization is not supported.')
                print('  specified gpu device: {}.'.format(gpu_device))
                sys.exit(1)
        
        self.impl = getattr(pylibheom, impl_class_name)()
        
        if (not gpu_device is None):
            self.impl.set_device_number(gpu_device)
        
        E, self.Z = np.linalg.eig(H)
        self.n_state = H.shape[0]
        self.impl.set_hamiltonian(get_coo_matrix(np.diag(E).astype(np.complex128)))
        

        n_noise = len(noises)
        self.impl.allocate_noises(n_noise)
        for u in range(n_noise):
            V = get_coo_matrix((self.Z.T.conj())@noises[u]["V"]@(self.Z).astype(np.complex128))
            if "func" in noises[u]["C"]:
                self.impl.set_noise_func(u, V, noises[u]["C"]["func"])
            else:    
                gamma   = noises[u]["C"]["gamma"]
                phi_0   = noises[u]["C"]["phi_0"]
                sigma   = noises[u]["C"]["sigma"]
                s       = noises[u]["C"]["s"]
                a       = noises[u]["C"]["a"]
                S_delta = noises[u]["C"]["S_delta"]
                self.impl.set_noise(u,
                                    V,
                                    get_coo_matrix(gamma.astype(np.complex128)),
                                    phi_0.astype(np.complex128),
                                    sigma.astype(np.complex128),
                                    get_coo_matrix(s.astype(np.complex128)),
                                    complex(S_delta),
                                    get_coo_matrix(a.astype(np.complex128)))
            
        
        self.rho = np.zeros((self.n_state, self.n_state),
                            dtype=np.complex128,
                            order='F')
        
        self.impl.init_aux_vars(callback,
                                callback_interval)
        
    def construct_commutator(self,
                             x, coef_l, coef_r,
                             callback=lambda lidx: None,
                             callback_interval=1024):
        x_coo = sp.sparse.coo_matrix((self.Z.T.conj())@x@(self.Z))
        self.impl.construct_commutator(x_coo.shape[0],
                                       x_coo.shape[1],
                                       x_coo.nnz,
                                       x_coo.row,
                                       x_coo.col,
                                       x_coo.data.astype(np.complex128),
                                       coef_l,
                                       coef_r,
                                       callback,
                                       callback_interval)

    def apply_commutator(self):
        self.impl.apply_commutator(self.rho.ravel())
        
    def set_rho(self, rho):
        self.rho[:,:] = (self.Z.T.conj())@rho[:,:]@(self.Z)

    def get_rho(self):
        return np.copy((self.Z)@self.rho[:,:]@(self.Z.T.conj()))

    def time_evolution(self, dt__unit, count,
                       callback=lambda t, rho: None,
                       callback_interval=1):
        self.impl.time_evolution(self.rho.ravel(order='F'),
                                 dt__unit, dt__unit*calc_unit(),
                                 callback_interval, count//callback_interval,
                                 lambda t: callback(t, (self.Z)@self.rho[:,:]@(self.Z.T.conj())))
