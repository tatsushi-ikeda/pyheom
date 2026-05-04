#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .solver_base import *
from os import environ
from multiprocessing import cpu_count

class HEOMSolver(QMESolver):
    """Hierarchical equations of motion (HEOM) solver.

    n_tiers controls the hierarchy truncation depth.
    """
    qme_name = 'heom'

    compulsory_args = [
        'n_tiers',
    ]

    optional_args = OrderedDict(
        n_inner_threads = lambda: int(environ.get('OMP_NUM_THREADS', cpu_count())),
        n_outer_threads = 1,
    )

    space_char = {
        'hilbert':   'h',
        'liouville': 'l',
        'ado':       'a',
    }

    def storage_size(self):
        return self.qme_impl.get_n_hrchy() + 1
