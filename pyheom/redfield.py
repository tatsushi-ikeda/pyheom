#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .solver_base import *

class RedfieldSolver(QMESolver):
    """Redfield master equation solver (Born-Markov approximation, no ADO hierarchy)."""
    qme_name = 'redfield'

    def storage_size(self):
        return 1
