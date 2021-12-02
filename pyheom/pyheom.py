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

version = getattr(pylibheom, 'version')()
__version__ = version

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
        print('[Error] Unsupported matrix type: {}.'.format(matrix.dtype),
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
                 hrchy_connection='loop',
                 hrchy_filter=None,
                 gpu_device=None,
                 callback=lambda lidx, est: None,
                 callback_interval=1024,
                 unrolling=False):
        self.n_state = H.shape[0]
        
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

        if   hrchy_connection == 'loop':
            impl_class_name += 'l'
        elif hrchy_connection == 'hierarchical-Liouville':
            impl_class_name += 'h'
        else:
            print('[Error] Unknown hrchy_connection: {}.'.format(
                hrchy_connection))
            sys.exit(1)

        if unrolling and self.n_state in [2, 3]:
            impl_class_name += '_{}'.format(self.n_state)
        
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
        
        self.impl.set_hamiltonian(get_coo_matrix(H.astype(np.complex128)))

        n_noise = len(noises)
        self.impl.alloc_noises(n_noise)
        
        self.noises = []
        
        for u in range(n_noise):
            gamma   = noises[u]["C"]["gamma"].astype(np.complex128)
            phi_0   = noises[u]["C"]["phi_0"].astype(np.complex128)
            sigma   = noises[u]["C"]["sigma"].astype(np.complex128)
            s       = noises[u]["C"]["s"].astype(np.complex128)
            a       = noises[u]["C"]["a"].astype(np.complex128)
            S_delta = complex(noises[u]["C"]["S_delta"])
            self.noises.append(type("noise", (object,),
                                    dict(gamma=gamma,
                                         phi_0=phi_0,
                                         sigma_s=s.T@sigma,
                                         sigma_a=a.T@sigma,
                                         S_delta=S_delta)))
            self.impl.set_noise(u,
                                get_coo_matrix(noises[u]["V"].astype(np.complex128)),
                                get_coo_matrix(gamma),
                                phi_0,
                                sigma,
                                get_coo_matrix(s),
                                S_delta,
                                get_coo_matrix(a))

        if hrchy_filter:
            self.hrchy_filter = lambda index, depth, lk: hrchy_filter(index, depth, lk, self.noises)
        else:
            self.hrchy_filter = lambda index, depth, lk, noises: True

        self.impl.linearize()
        self.n_hrchy \
            = self.impl.alloc_hrchy(max_tier,
                                    callback,
                                    callback_interval,
                                    self.hrchy_filter,
                                    False if hrchy_filter is None else True)
        self.rho_h = np.zeros((self.n_state, self.n_state, self.n_hrchy),
                              dtype=np.complex128, order='F')
        
        self.impl.init_aux_vars()
        
    def construct_commutator(self,
                             x, coef_l, coef_r,
                             callback=lambda lidx, est: None,
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
        self.impl.apply_commutator(self.rho_h.ravel(order='F'))

    def set_rho(self, rho):
        self.rho_h[:,:,0] = rho[:,:]

    def get_rho(self):
        return np.copy(self.rho_h[:,:,0])

    def set_rho_h(self, rho_h):
        self.rho_h[:,:,:] = rho_h[:,:,:]

    def get_rho_h(self):
        return np.copy(self.rho_h[:,:,:])

    def calc_diff(self, rho_h):
        drho_h_dt = np.zeros_like(rho_h)
        self.impl.calc_diff(drho_h_dt.ravel(order='F'),
                            rho_h.ravel(order='F'),
                            1, 0)
        return drho_h_dt

    def get_diff_func(self):
        return lambda t, rho_h: self.calc_diff(rho_h)

    def solve(self, dt__unit, count,
                       callback=lambda t, rho: None,
                       callback_interval=1):
        self.impl.solve(self.rho_h.ravel(order='F'),
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
                 callback_interval=1024,
                 unrolling=False,
                 secular=False,
                 H_c=None):
        self.n_state = H.shape[0]
        
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
        
        if unrolling and self.n_state in [2, 3]:
            impl_class_name += '_{}'.format(self.n_state)
        
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
        self.impl.set_hamiltonian(get_coo_matrix(np.diag(E).astype(np.complex128)))
        if H_c is None:
            H_c = np.zeros_like(H)
        
        self.impl.set_redfield_options(get_coo_matrix(self.Z.T.conj()@H_c@(self.Z).astype(np.complex128)),
                                       secular)

        n_noise = len(noises)
        self.impl.alloc_noises(n_noise)
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
        
        self.impl.init_aux_vars()
        
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
        self.impl.apply_commutator(self.rho.ravel(order='F'))
        
    def set_rho(self, rho):
        self.rho[:,:] = (self.Z.T.conj())@rho[:,:]@(self.Z)

    def get_rho(self):
        return np.copy((self.Z)@self.rho[:,:]@(self.Z.T.conj()))

    def calc_diff(self, rho):
        drho_dt = np.zeros_like(rho)
        self.impl.calc_diff(drho_dt.ravel(order='F'),
                            ((self.Z.T.conj())@rho.reshape((self.n_state, self.n_state), order='F')@(self.Z)).ravel(order='F'),
                            1, 0)
        return ((self.Z)@drho_dt.reshape((self.n_state, self.n_state), order='F')@(self.Z.T.conj())).ravel(order='F')
    
    def get_diff_func(self):
        return lambda t, rho: self.calc_diff(rho)

    def solve(self, dt__unit, count,
                       callback=lambda t, rho: None,
                       callback_interval=1):
        self.impl.solve(self.rho.ravel(order='F'),
                        dt__unit, dt__unit*calc_unit(),
                        callback_interval, count//callback_interval,
                        lambda t: callback(t, (self.Z)@self.rho[:,:]@(self.Z.T.conj())))
