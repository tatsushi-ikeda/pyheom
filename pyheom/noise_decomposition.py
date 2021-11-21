# 
# LibHEOM: Copyright (c) Tatsushi Ikeda
# This library is distributed under BSD 3-Clause License.
# See LINCENSE.txt for licence.
# ------------------------------------------------------------------------

import numpy as np
import scipy as sp
import scipy.sparse
import itertools
from collections import OrderedDict

from .predefined_noise            import *
from .summation_over_poles        import *
from .commuting_matrix            import *
from .pade_spectral_decomposition import *

fsd_coeffs = {
    100.0:  [[1,  1.35486,   1.34275],
             [2,  5.50923,   0.880362],
             [3,  0.553793, -0.965783]],
    1000.0: [[1, 79.1394,    0.637942],
             [1,  0.655349,  0.666920],
             [2, 11.6632,    0.456271],
             [2,  1.54597,   0.740457],
             [3,  3.39011,   0.626892]],
}

def calc_S_MSD(gamma_k, a_k, T, n_ltc):
    def cot(x):
        return 1/np.tan(x)
    
    n_m = a_k.shape[0]
    n_k = n_ltc
    
    nu_k    = np.zeros(n_k)
    s_k     = np.zeros(n_m + n_k, dtype=a_k.dtype)
    S_delta = 0.0
    
    for k in range(n_k):
        nu_k[k] = 2*np.pi*(k + 1)*T
        if np.any(np.abs(gamma_k - nu_k[k]) < (np.finfo(float).eps)):
            raise Exception('[Error] Bath frequency #{} is degenerated.'.format(k))

    # r_k[m] --> 
    for m in range(n_m):
        s_k[m] = -2*a_k[m]*cot(gamma_k[m]/(2*T))/2.0

    for k in range(n_k):
        s_k[n_m+k] = 0.0
        for m in range(n_m):
            s_k[n_m+k] += -4*a_k[m]/(nu_k[k]**2 - gamma_k[m]**2)
        s_k[n_m+k] *= T*nu_k[k]

    for m in range(n_m):
        inner = 1/gamma_k[m]**2 - 1/(2*T*gamma_k[m])*cot(gamma_k[m]/(2*T))
        for k in range(n_k):
            inner -= 2/(nu_k[k]**2 - gamma_k[m]**2)
        S_delta += -2*T*a_k[m]*inner
    
    result = OrderedDict()

    def put_coeff(a, m, coeff):
        if (a, m) in result:
            result[(a, m)] += coeff
        else:
            result[(a, m)] = coeff

    put_coeff(np.inf, 0, S_delta)
    for k in range(n_m):
        put_coeff(gamma_k[k], 0, s_k[k])

    for k in range(n_k):
        put_coeff(nu_k[k], 0, s_k[k + n_m])
    
    return result

def calc_noise_time_domain(J, T, type_LTC, **kwargs):
    if (type_LTC == 'NONE'):
        n_list = [[0, T, 1, 0]]

        return calc_S_from_poles(J.poles, n_list), calc_A_from_poles(J.poles)
    
    elif (type_LTC == 'MSD'):
        n_MSD = kwargs['n_MSD']
        
        A = calc_A_from_poles(J.poles)
        
        gamma_k = np.zeros(len(A), dtype=np.complex128)
        a_k     = np.zeros(len(A), dtype=np.complex128)
        for k, (gamma, l) in enumerate(A.keys()):
            if l != 0:
                raise Exception('[Error] MSD accepts only first-order poles')
            gamma_k[k] = gamma
            a_k[k]     = A[(gamma, 0)]
            
        return calc_S_MSD(gamma_k, a_k, T, n_MSD), A
    
    elif (type_LTC == 'PSD'  or type_LTC == 'PSD+FSD' ):
        coeffs = []
        coeff_0 = 0
        
        if type_LTC == 'PSD+FSD':
            n_FSD_rec = kwargs['n_FSD_rec']
            chi_FSD   = kwargs['chi_FSD']
            # calc FSD coeffs
            T_n = T
            for i in range(n_FSD_rec):
                T_np1 = T_n*chi_FSD
                coeff_0 += T_n - T_np1
                T_n   = T_np1
                for j, a, b in fsd_coeffs[chi_FSD]:
                    coeffs.append([j, a, b, T_n])
            T_0 = T_n
        else:
            T_0 = T
        
        # calc PSD coeffs
        n_PSD    = kwargs['n_PSD']
        type_PSD = kwargs['type_PSD']
        xi, eta, R_1, T_3 = PSD(n_PSD, type_PSD)
        
        # collect poles
        poles = OrderedDict()
        ## PSD poles
        poles[(0, 1, 0)] = T_0
        if (R_1 != 0):
            poles[(0, 0, 0)] = R_1
        if (T_3 != 0):
            poles[(0, 0, 1)] = T_3
        for p in range(n_PSD):
            poles[(T_0*xi[p], 1, 0)] = 2*eta[p]*T_0
        
        ## FSD poles
        poles[(0, 1, 0)] += coeff_0
        for j, a, b, T_n in coeffs:
            poles[(T_n/a, j, 0)] = b*(T_n/a)**(2*j-1)
        
        n_list = [[a, b, m, n] for (a, m, n), b in poles.items()]
        print(n_list)
        
        return calc_S_from_poles(J.poles, n_list), calc_A_from_poles(J.poles)
    else:
        raise Exception('[Error] Unknown LTC')


def calc_noise_params(S, A):

    # Calculate Basis Degeneracy
    phi_deg_dict = OrderedDict()
    for gamma, n in itertools.chain(S.keys(), A.keys()):
        if (gamma == np.inf):
            continue
        if gamma in phi_deg_dict:
            phi_deg_dict[gamma] = max(phi_deg_dict[gamma], n + 1)
        else:
            phi_deg_dict[gamma] = n + 1
    phi_dim = sum((n for n in phi_deg_dict.values()))
    
    # 
    phi   = []
    phi_0 = np.zeros((phi_dim), np.complex128)
    gamma = sp.sparse.lil_matrix((phi_dim, phi_dim), dtype=np.complex128)
    sigma = np.ones((phi_dim), np.complex128)
    
    s_vec = np.zeros((phi_dim), np.complex128)
    a_vec = np.zeros((phi_dim), np.complex128)
    s_mat = sp.sparse.lil_matrix((phi_dim, phi_dim), dtype=np.complex128)
    a_mat = sp.sparse.lil_matrix((phi_dim, phi_dim), dtype=np.complex128)
    
    ctr   = 0
    for gamma_n, deg_max in phi_deg_dict.items():
        for deg in range(deg_max):
            phi.append((gamma_n, deg))
            phi_0[ctr] = 1 if deg == 0 else 0
            gamma[ctr,ctr] = gamma_n
            if deg > 0:
                gamma[ctr,ctr-1] = -deg
            if ((gamma_n, deg) in S):
                s_vec[ctr] = S[(gamma_n, deg)]
            if ((gamma_n, deg) in A):
                a_vec[ctr] = A[(gamma_n, deg)]
            ctr += 1
        block_size = deg+1
        s_mat[ctr-block_size:ctr, ctr-block_size:ctr] \
            = get_commuting_matrix(s_vec[ctr-block_size:ctr],
                                   gamma[ctr-block_size:ctr, ctr-block_size:ctr].todense(),
                                   sigma[ctr-block_size:ctr])
        a_mat[ctr-block_size:ctr, ctr-block_size:ctr] \
            = get_commuting_matrix(a_vec[ctr-block_size:ctr],
                                   gamma[ctr-block_size:ctr, ctr-block_size:ctr].todense(),
                                   sigma[ctr-block_size:ctr])
    S_delta = 0.0
    if (np.inf, 0) in S:
        S_delta = S[(np.inf, 0)]
    
    return dict(gamma   = gamma,
                sigma   = sigma,
                phi_0   = phi_0,
                s       = s_mat,
                S_delta = S_delta,
                a       = a_mat)

def noise_decomposition(J, T, type_LTC, **kwargs):
    return calc_noise_params(*calc_noise_time_domain(J, T, type_LTC, **kwargs))

# noise = calc_noise_params(*calc_noise_time_domain(None, T, 'PSD+FSD', n_PSD = 1, type_PSD = 'N/N', n_FSD_rec=1, chi_FSD=100.0))
# noise = calc_noise_params(*calc_noise_time_domain(J, T, 'PSD+FSD',
#                                                   n_PSD = 1, type_PSD = 'N/N',
#                                                   n_FSD_rec=1, chi_FSD=100.0))
# noise = calc_noise_params(*calc_noise_time_domain(J, T, 'PSD',
#                                                   n_PSD = 1, type_PSD = 'N-1/N'))
# noise = calc_noise_params(*calc_noise_time_domain(J, T, 'MSD',
#                                                   n_MSD = 10))
# noise = calc_noise_params(*calc_noise_time_domain(J, T, 'NONE'))
