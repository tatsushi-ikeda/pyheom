#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np
import scipy as sp
import scipy.special
from collections import OrderedDict

def calc_a_from_poles(poles):
    """Compute A(t) = -1/pi integral_0^inf f(omega) sin(omegat) domega analytically from rational poles.

    Each pole entry is [a, b, m, n] encoding f(omega) += b omega^(2n+1) / (a^2+omega^2)^m.
    Returns OrderedDict {(a, l): c} representing A(t) = Sigma c t^l exp(-a t).
    """
    result = OrderedDict()
    
    def put_coeff(a, m, coeff):
        if (a, m) in result:
            result[(a, m)] += coeff
        else:
            result[(a, m)] = coeff
            
    for j in range(len(poles)):
        a, b, m, n = poles[j]
        
        def sub(a, b, m, n):
            for l in range(m):
                inner1 = 0
                for p in range(min(m - l, 2*n + 2)):
                    inner1 += (sp.special.binom(m - l - 1, p)
                               *sp.special.poch(m, m - l - p - 1)
                               *sp.special.poch(2*(n + 1) - p, p)
                               *(0.5)**((2*m - l - p - 1))
                               *(-1)**(p + n + 1))
                put_coeff(a, l,
                          (b/sp.special.factorial(m - 1)
                           *sp.special.binom(m - 1, l)
                           *a**(2*(n - m + 1) + l)
                          *inner1))
                
        sub(a, b, m, n)
    return result


def calc_s_from_poles(poles_1, poles_2):
    """Compute S(t) = 2/pi integral_0^inf f(omega) g(omega) cos(omegat) domega analytically from rational poles.

    poles_1 encodes f (spectral density poles); poles_2 encodes g (Bose-Einstein or LTC poles).
    Returns OrderedDict {(a, l): c}; entries with a=inf represent delta-function contributions.
    """

    result = OrderedDict()
    
    def put_coeff(a, m, coeff):
        if (a, m) in result:
            result[(a, m)] += coeff
        else:
            result[(a, m)] = coeff

    for a_, b_, m_, n_ in poles_2:
        for a, b, m, n in poles_1:
            if (a_ == 0 or a == a_):
                def sub(a, b, b_, M, N):
                    if (N == M + 1):
                        put_coeff(np.inf, 2, -b*b_)
                        put_coeff(np.inf, 0, -b*b_*a**2*M)
                        for l in range(M):
                            inner1 = 0
                            for r in range(M):
                                inner2 = 0
                                for p in range(min(M - l, 2*r + 1)):
                                    inner2 += (sp.special.binom(M - l - 1, p)
                                               *sp.special.poch(M, M - l - p - 1)
                                               *sp.special.poch(2*r - p + 1, p)
                                               *(0.5)**(2*M - l - p - 1)
                                               *(-1)**(r - p + 1))
                                inner1 += (M - r)*sp.special.binom(M + 1, r)*inner2
                            put_coeff(a, l,
                                      -sp.special.binom(M - 1, l)
                                      *2*b*b_/sp.special.factorial(M - 1)
                                      *a**(l + 3)
                                      *inner1)
                    elif (N == M):
                        put_coeff(np.inf, 0, b*b_)
                        for l in range(M):
                            inner1 = 0
                            for r in range(M):
                                inner2 = 0
                                for p in range(min(M - l, 2*r + 1)):
                                    inner2 += (sp.special.binom(M - l - 1, p)
                                               *sp.special.poch(M, M - l - p - 1)
                                               *sp.special.poch(2*r - p + 1, p)
                                               *(0.5)**(2*M - l - p - 1)
                                               *(-1)**(r - p + 1))
                                inner1 += sp.special.binom(M, r)*inner2
                            put_coeff(a, l,
                                      sp.special.binom(M - 1, l)
                                      *2*b*b_/sp.special.factorial(M - 1)
                                      *a**(l + 1)
                                      *inner1)
                    elif (N < M):
                        for l in range(M):
                            inner1 = 0
                            for p in range(min(M - l, 2*N + 1)):
                                inner1 += (sp.special.binom(M - l - 1, p)
                                           *sp.special.poch(M, M - l - p - 1)
                                           *sp.special.poch(2*N - p + 1, p)
                                           *(0.5)**(2*M - l - p - 1)
                                           *(-1)**(N - p))
                            put_coeff(a, l,
                                      sp.special.binom(M - 1, l)
                                      *2*b*b_/sp.special.factorial(M - 1)
                                      *a**(2*(N - M) + l + 1)
                                      *inner1)
                    else:
                        raise ValueError('An invalid pole is given to calc_s_from_poles')
                if (a == a_):
                    sub(a, b, b_, m + m_, n + n_ + 1)
                else:
                    if   (m_ == 1 and n_ == 0):
                        sub(a, b, b_, m, n)
                    elif (m_ == 0 and n_ == 0):
                        sub(a, b, b_, m, n + 1)
                    elif (m_ == 0 and n_ == 1):
                        sub(a, b, b_, m, n + 2)
                    else:
                        raise ValueError('An invalid pole is given to calc_s_from_poles')
            else:
                def sub(a, b, m, n, a_, b_, m_, n_):
                    for l in range(m):
                        inner1 = 0
                        for p in range(min(m - l, 2*(n + n_ + 1) + 1)):
                            inner2 = 0
                            for q in range(m - l - p):
                                inner3 = 0
                                for r in range(q + 1):
                                    inner3 += (sp.special.binom(q, r)
                                               /((a - a_)**r*(a + a_)**(q - r))
                                               *sp.special.poch(m_, q - r)
                                               *sp.special.poch(m_, r)
                                               /(-a**2 + a_**2)**m_)
                                inner2 += (sp.special.binom(m - l - p - 1, q)
                                           *sp.special.poch(m, m - l - p - q - 1)
                                           /(2*a)**(2*m - l - p - q - 1)
                                           *inner3)
                            inner1 += (sp.special.binom(m - l - 1, p)
                                       *(-1)**(n + n_ - p - 1)
                                       *sp.special.poch(2*(n + n_ + 1) - p + 1, p)
                                       *(a)**(2*(n + n_ + 1) - p)
                                       *inner2)
                        put_coeff(a, l,
                                  2*b*b_
                                  /sp.special.factorial(m - 1)
                                  *sp.special.binom(m - 1, l)
                                  *inner1)
                sub(a, b, m, n, a_, b_, m_, n_)
                sub(a_, b_, m_, n_, a, b, m, n)
    return result
