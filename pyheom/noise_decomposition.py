#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np
import scipy as sp
import scipy.sparse
import itertools
from dataclasses import dataclass, field

from .spectral_density     import *
from .summation_over_poles import *
from .commuting_matrix     import *
from .pade_decomposition   import *


_FSD_COEFFS = {
    100.0:  [[1,  1.35486,   1.34275],
             [2,  5.50923,   0.880362],
             [3,  0.553793, -0.965783]],
    1000.0: [[1, 79.1394,    0.637942],
             [1,  0.655349,  0.666920],
             [2, 11.6632,    0.456271],
             [2,  1.54597,   0.740457],
             [3,  3.39011,   0.626892]],
}


@dataclass
class BathCorrelation:
    """Bath correlation function decomposed into K exponential modes.

    Returned by `noise_decomposition`, or constructed directly for a known
    decomposition C(t) = sum_k (s_k - i*a_k) exp(-gamma_k t) + s_delta*delta(t).
    Set `V` before passing to a solver.  See `docs/api.md` for construction details.
    """
    gamma:   object         # scipy.sparse (K,K)
    sigma:   np.ndarray     # (K,)
    phi_0:   np.ndarray     # (K,)
    s_mat:   object         # scipy.sparse (K,K)
    a_mat:   object         # scipy.sparse (K,K)
    s_delta: complex = 0.0
    V:       object = None  # set by user


def calc_s_msd(gamma_k, a_k, T, n_ltc):
    def cot(x):
        return 1/np.tan(x)
    
    n_m = a_k.shape[0]
    n_k = n_ltc
    
    nu_k    = np.zeros(n_k)
    s_k     = np.zeros(n_m + n_k, dtype=a_k.dtype)
    s_delta = 0.0
    
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
        s_delta += -2*T*a_k[m]*inner
    
    result = {}

    result[(np.inf, 0)] = s_delta
    for k in range(n_m):
        result[(gamma_k[k], 0)] = result.get((gamma_k[k], 0), 0) + s_k[k]
    for k in range(n_k):
        result[(nu_k[k], 0)] = result.get((nu_k[k], 0), 0) + s_k[k + n_m]
    
    return result

def calc_bath_corr_poles(J, T, type_ltc, **kwargs):
    """Return (S, A) pole dicts for the symmetric/antisymmetric bath correlation functions.

    type_ltc: 'none' (no LTC), 'msd' (Matsubara), 'psd' (Pade), or 'psd+fsd'.
    """
    type_ltc = type_ltc.lower()
    
    if (type_ltc == 'none'):
        n_list = [[0, T, 1, 0]]

        return calc_s_from_poles(J.poles, n_list), calc_a_from_poles(J.poles)
    
    elif (type_ltc == 'msd'):
        n_msd = kwargs['n_msd']
        
        A = calc_a_from_poles(J.poles)
        
        gamma_k = np.zeros(len(A), dtype=np.complex128)
        a_k     = np.zeros(len(A), dtype=np.complex128)
        for k, (gamma, l) in enumerate(A.keys()):
            if l != 0:
                raise Exception('[Error] msd accepts only first-order poles')
            gamma_k[k] = gamma
            a_k[k]     = A[(gamma, 0)]
            
        return calc_s_msd(gamma_k, a_k, T, n_msd), A
    
    elif (type_ltc == 'psd'  or type_ltc == 'psd+fsd' ):
        coeffs = []
        coeff_0 = 0
        
        if type_ltc == 'psd+fsd':
            n_fsd_rec = kwargs['n_fsd_rec']
            chi_fsd   = kwargs['chi_fsd']
            # calc fsd coeffs
            T_n = T
            for i in range(n_fsd_rec):
                T_np1 = T_n*chi_fsd
                coeff_0 += T_n - T_np1
                T_n   = T_np1
                for j, a, b in _FSD_COEFFS[chi_fsd]:
                    coeffs.append([j, a, b, T_n])
            T_0 = T_n
        else:
            T_0 = T
        
        # calc psd coeffs
        n_psd    = kwargs['n_psd']
        type_psd = kwargs['type_psd']
        xi, eta, R_1, T_3 = psd(n_psd, type_psd)
        
        # collect poles
        poles = {}
        ## psd poles
        poles[(0, 1, 0)] = T_0
        if (R_1 != 0):
            poles[(0, 0, 0)] = R_1
        if (T_3 != 0):
            poles[(0, 0, 1)] = T_3
        for p in range(n_psd):
            poles[(T_0*xi[p], 1, 0)] = 2*eta[p]*T_0
        
        ## fsd poles
        poles[(0, 1, 0)] += coeff_0
        for j, a, b, T_n in coeffs:
            poles[(T_n/a, j, 0)] = b*(T_n/a)**(2*j-1)
        
        n_list = [[a, b, m, n] for (a, m, n), b in poles.items()]
        
        return calc_s_from_poles(J.poles, n_list), calc_a_from_poles(J.poles)
    else:
        raise Exception('[Error] Unknown ltc')


def _poles_to_bath_corr(S, A):
    """Convert (S, A) pole dicts to a BathCorrelation via the commuting-matrix construction."""

    # Calculate Basis Degeneracy
    phi_deg_dict = {}
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
    
    idx   = 0
    for gamma_n, deg_max in phi_deg_dict.items():
        for deg in range(deg_max):
            phi.append((gamma_n, deg))
            phi_0[idx] = 1 if deg == 0 else 0
            gamma[idx,idx] = gamma_n
            if deg > 0:
                gamma[idx,idx-1] = -deg
            if ((gamma_n, deg) in S):
                s_vec[idx] = S[(gamma_n, deg)]
            if ((gamma_n, deg) in A):
                a_vec[idx] = A[(gamma_n, deg)]
            idx += 1
        block_size = deg+1
        s_mat[idx-block_size:idx, idx-block_size:idx] \
            = get_commuting_matrix(s_vec[idx-block_size:idx],
                                   gamma[idx-block_size:idx, idx-block_size:idx].todense(),
                                   sigma[idx-block_size:idx])
        a_mat[idx-block_size:idx, idx-block_size:idx] \
            = get_commuting_matrix(a_vec[idx-block_size:idx],
                                   gamma[idx-block_size:idx, idx-block_size:idx].todense(),
                                   sigma[idx-block_size:idx])
    s_delta = 0.0
    if (np.inf, 0) in S:
        s_delta = S[(np.inf, 0)]

    return BathCorrelation(
        gamma   = gamma,
        sigma   = sigma,
        phi_0   = phi_0,
        s_mat   = s_mat,
        s_delta = s_delta,
        a_mat   = a_mat,
    )

def noise_decomposition(J, T, type_ltc, **kwargs):
    """Decompose the bath correlation function of J into exponential modes.

    Parameters
    ----------
    J : SpectralDensity
        Spectral density model (e.g. `Drude`, `Brown`).
    T : float
        Temperature.
    type_ltc : {'none', 'msd', 'psd', 'psd+fsd'}
        Low-temperature correction method.
    n_msd : int, optional
        Number of Matsubara terms; required when `type_ltc='msd'`.
    n_psd : int, optional
        Number of Pade poles; required when `type_ltc='psd'` or `'psd+fsd'`.
    type_psd : str, optional
        Pade variant (e.g. `'N-1/N'`, `'N/N'`); required with `n_psd`.
    n_fsd_rec : int, optional
        Number of FSD recursion levels; required when `type_ltc='psd+fsd'`.
    chi_fsd : float, optional
        FSD scaling factor (`100.0` or `1000.0`); required when `type_ltc='psd+fsd'`.

    Returns
    -------
    BathCorrelation
        Set `.V` (system-bath coupling) before passing to a solver.

    Examples
    --------
    >>> corr = noise_decomposition(J, T, 'none')
    >>> corr = noise_decomposition(J, T, 'msd', n_msd=10)
    >>> corr = noise_decomposition(J, T, 'psd', n_psd=1, type_psd='N-1/N')
    >>> corr = noise_decomposition(J, T, 'psd+fsd', n_psd=1, type_psd='N/N',
    ...                            n_fsd_rec=1, chi_fsd=100.0)
    """
    return _poles_to_bath_corr(*calc_bath_corr_poles(J, T, type_ltc, **kwargs))
