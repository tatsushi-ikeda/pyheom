#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .qme_solver import *
from os import environ
from multiprocessing import cpu_count

class heom_solver(qme_solver):
    """Hierarchical equations of motion (HEOM) solver.

    n_tiers controls the hierarchy truncation depth.
    """
    qme_name = 'heom'

    compulsory_args = [
        'n_tiers',
    ]
    
    optional_args = OrderedDict(
        n_inner_threads = 1,
        # U1: evaluated at construction time (not at module import) via callable default
        n_outer_threads = lambda: int(environ.get('OMP_NUM_THREADS', cpu_count())),
    )
    
    space_char = {
        'hilbert':   'h',
        'liouville': 'l',
        'ado':       'a',
    }

    def storage_size(self):
        return self.qme_impl.get_n_hrchy() + 1

