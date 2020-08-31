# 
# LibHEOM, version 0.5
# Copyright (c) 2019-2020 Tatsushi Ikeda
#
# This library is distributed under BSD 3-Clause License.
# See LINCENSE.txt for licence.
# ------------------------------------------------------------------------

import numpy as np
import scipy as sp

def get_commuting_matrix(c_vec, gamma, sigma):
    N = gamma.shape[0]
    if N == 1:
        return np.array([[c_vec[0]]])
    basis = np.zeros((N,N), dtype=np.complex128)
    gammaT_n = np.identity(N, dtype=np.complex128) # gamma**n
    for n in range(N):
        basis[:,n] = basis[:,n] + (gammaT_n@sigma)
        gammaT_n = gammaT_n.dot(gamma.T)
    # basis_inv = np.linalg.inv(basis)
    basis_inv = np.linalg.solve(basis, np.identity(N, dtype=np.complex128))
    coeff = basis_inv@c_vec
    
    gamma_n = np.identity(N, dtype=np.complex128)
    c_mat = np.zeros((N,N), dtype=np.complex128)
    for n in range(N):
        c_mat += coeff[n]*gamma_n
        gamma_n = gamma_n@gamma
    return c_mat

def get_commuting_matrix_diag(c_vec, gamma, sigma):
    N = gamma.shape[0]
    if N == 1:
        return np.array([[c_vec[0]]])

    s, U = sp.linalg.eig(gamma)
    Uinv = sp.linalg.inv(U)
    # print(Uinv@gamma@U)
    Delta = np.zeros((N,N), dtype=np.complex128)
    for j in range(N):
        Delta[j,j] = (c_vec@U[:,j])/(sigma@U[:,j])
    c_mat = U@Delta@Uinv
    return c_mat
