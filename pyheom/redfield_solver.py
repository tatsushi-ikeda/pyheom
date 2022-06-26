#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .qme_solver import *

class redfield_solver(qme_solver):
    qme_name = 'redfield'

    def storage_size(self):
        return 1
