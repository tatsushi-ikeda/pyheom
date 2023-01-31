#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .qme_solver import *
from os import environ
from multiprocessing import cpu_count

class heom_solver(qme_solver):
    qme_name = 'heom'

    compulsory_args = [
        'depth',
    ]
    
    optional_args = OrderedDict(
        n_inner_threads = 1,
        n_outer_threads = int(environ.get('OMP_NUM_THREADS', cpu_count())),
    )
    
    space_char = {
        'hilbert':   'h',
        'liouville': 'l',
        'ado':       'a',
    }

    def storage_size(self):
        return self.qme_impl.get_n_hrchy() + 1

