# 
# LibHEOM, version 0.5
# Copyright (c) 2019-2020 Tatsushi Ikeda
#
# This library is distributed under BSD 3-Clause License.
# See LINCENSE.txt for licence.
# ------------------------------------------------------------------------

import numpy as np
import cmath as cm

def psd(n, type_pade):
    if   (type_pade == 'N/N'):
        return psd_n(n)
    elif (type_pade == 'N-1/N'):
        return psd_nm1(n)
    elif (type_pade == 'N+1/N'):
        return psd_np1(n)
    else:
        raise Exception('[Error] Undefined type_pade: {}'.format(type_pade))

def psd_n(n, dtype=np.float64):
    if n == 0:
        return np.zeros(0), np.zeros(0), 1/12.0, 0.0
    
    m = 2*n + 1
    b = np.array([2*i + 3 for i in range(m)], dtype=np.float64)
    
    Lambda = np.zeros((m, m), dtype=np.float64)
    for i in range(m - 1):
        Lambda[i,i+1] = 1.0/np.sqrt(b[i]*b[i+1])
        Lambda[i+1,i] = Lambda[i,i+1]
    
    lambda_eig, _ = np.linalg.eigh(Lambda)
    lambda_eig.sort()
    xi = np.array([-2.0/l for l in lambda_eig[0:n]])
    
    Lambda_ = np.zeros((m-1, m-1), dtype=np.float64)
    for i in range(m - 2):
        Lambda_[i,i+1] = 1.0/np.sqrt(b[i+1]*b[i+2])
        Lambda_[i+1,i] = Lambda_[i,i+1]
    lambda_eig_, _ = np.linalg.eigh(Lambda_)
    lambda_eig_.sort()

    zeta = np.array([-2.0/l for l in lambda_eig_[0:n]])

    R = 1/(4*(n + 1)*b[n])

    eta = np.empty(n)
    for i in range(n):
        eta[i] = R/2.0;
        for k in range(n):
            eta[i] *= zeta[k]**2 - xi[i]**2
            if k != i:
                eta[i] /= xi[k]**2 - xi[i]**2

    T = 0
    
    return xi, eta, R, T


def psd_nm1(n):
    if (n == 0):
        raise
    
    m = 2*n
    b = np.array([2*i + 3 for i in range(m)], dtype=np.float64)
    
    Lambda = np.zeros((m, m), dtype=np.float64)
    for i in range(m - 1):
        Lambda[i,i+1] = 1.0/np.sqrt(b[i]*b[i+1])
        Lambda[i+1,i] = Lambda[i,i+1]
    
    lambda_eig, _ = np.linalg.eigh(Lambda)
    lambda_eig.sort()
    xi_ = np.array([-2.0/l for l in lambda_eig[0:n]])
    
    Lambda_ = np.zeros((m-1, m-1), dtype=np.float64)
    for i in range(m - 2):
        Lambda_[i,i+1] = 1.0/np.sqrt(b[i+1]*b[i+2])
        Lambda_[i+1,i] = Lambda_[i,i+1]
    lambda_eig_, _ = np.linalg.eigh(Lambda_)
    lambda_eig_.sort()

    zeta_ = np.array([-2.0/l for l in lambda_eig_[0:n]])

    R_ = 0.0
    
    eta_ = np.empty(n)
    for i in range(n):
        eta_[i] = n*b[n]/2.0;
        for k in range(n):
            if k != n - 1:
                eta_[i] *= zeta_[k]**2 - xi_[i]**2
            if k != i:
                eta_[i] /= xi_[k]**2 - xi_[i]**2

    T_ = 0.0

    return xi_, eta_, R_, T_


def psd_np1(N):
    if (N == 0):
        raise
    
    M = 2*N + 2
    
    # calc b[m]
    b = np.array([2*m + 3 for m in range(M)], dtype=np.float64)

    # calc d[m]
    d = np.zeros((M), dtype=np.float64)
    d[0] = 1/(4*b[0])
    for m in range(1,M//2+1):
        d[2*m-1]  = -4*m**2*b[m-1]**2*b[2*m-1]
    for m in range(1,M//2):
        d[2*m] = -b[2*m]/(4*m*(m+1)*b[m-1]*b[m])

    # calc T_caron[k]
    T_caron   = np.zeros((N+1), dtype=np.float64)
    for k in range(0,N+1):
        denom = 0.0
        for n in range(1,k+2):
            denom += d[2*n-1]
        T_caron[k]=1/(4*denom)
    T_caron_N = T_caron[N]

    # calc R_caron[N]
    summation = 0.0
    for m in range(1,N+2):
        inner_summation = 0.0
        for k in range(m, N+2):
            inner_summation += d[2*k-1]
        summation += d[2*m-2]*inner_summation**2
    R_caron_N = (4*T_caron[N])**2*summation

    # calc xi_caron
    Lambda_prime = np.zeros((M, M), dtype=np.float64)
    for n in range(1,M):
        m = n+1
        if (m <= M-1):
            Lambda_prime[m,n] = 1.0/np.sqrt(d[m]*d[n])
        m = n-1
        if (m >= 1):
            Lambda_prime[m,n] = 1.0/np.sqrt(d[m]*d[n])
    lambda_prime_eig, _ = np.linalg.eigh(Lambda_prime[1:M,1:M])
    lambda_prime_eig.sort()
    xi_caron = np.zeros((N+1), dtype=np.float64)
    xi_caron[1:N+1] = np.array([-2.0/l for l in lambda_prime_eig[0:N]])


    # calc t[k]
    t  = np.zeros((N+2), dtype=np.float64)
    t[1] = T_caron[0]
    for k in range(1,N+1):
        t[k+1] = T_caron[k]/T_caron[k-1]
    
    # calc eta_caron
    eta_caron = np.zeros((N+1), dtype=np.complex64)
    for j in range(1,N+1):
        delta = np.zeros((N+2), dtype=np.complex128)
        for k in range(1,N+1):
            delta[k] = xi_caron[k]**2 - xi_caron[j]**2
        delta[j] = 1.0
        delta[N+1] = 1.0
        r = np.zeros((2*N+3), dtype=np.complex128)
        for k in range(1,N+2):
            r[2*k-1] = cm.sqrt(4*np.abs(t[k]/delta[k]))
            r[2*k] = r[2*k-1]*np.sign(t[k]/delta[k])
        X = np.zeros((2*N+3), dtype=np.complex128)
        X[0] = 0.5
        X[1] = d[0]*r[1]*X[0]
        for m in range(2,2*N+3):
            X[m] = d[m-1]*r[m]*X[m-1] - r[m]*r[m-1]*xi_caron[j]**2*X[m-2]/4
        eta_caron[j] = X[2*N+2]

    return xi_caron[1:], eta_caron[1:], R_caron_N, T_caron_N
