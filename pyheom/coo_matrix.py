#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np
import scipy as sp
import scipy.sparse
import pyheom.pylibheom as libheom

from .const import *

def libheom_coo_matrix(m):
    if isinstance(m, list):
        m = np.array(m)
    if m.dtype in DTYPE_CMPLX.keys():
        impl_class_name = f'coo_matrix_{DTYPE_CHAR[DTYPE_CMPLX[m.dtype]]}'
    else:
        raise TypeError(f'Unsupported matrix type: {m.dtype}')
    coo = sp.sparse.coo_matrix(m)
    dtype = DTYPE_CMPLX[m.dtype]
    return getattr(libheom, impl_class_name)(
        coo.shape[0],
        coo.shape[1],
        coo.nnz,
        coo.row,
        coo.col,
        coo.data.astype(dtype)
    )
