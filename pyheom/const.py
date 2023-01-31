#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np

DTYPE_CHAR = {
    np.dtype('complex64'):  'c',
    np.dtype('complex128'): 'z',
}

DTYPE_CMPLX = {
    np.dtype('float32'):    np.dtype('complex64'),
    np.dtype('complex64'):  np.dtype('complex64'),
    np.dtype('int64'):      np.dtype('complex128'),
    np.dtype('float64'):    np.dtype('complex128'),
    np.dtype('complex128'): np.dtype('complex128'),
}

FORMAT_CHAR = {
    'dense':  'd',
    'sparse': 's',
}

ORDER_CHAR  = {
    'row_major': 'r',
    'col_major': 'c',
    True:  'r',
    False: 'c',
}

ORDER_CHAR_NUMPY  = {
    'row_major': 'C',
    'col_major': 'F',
    True:  'C',
    False: 'F',
}

ENGINE_ARGS = {
    'eigen': {},
    'mkl':   {},
    'cuda':  {'device':0},
}



