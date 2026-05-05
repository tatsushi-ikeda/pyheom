#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .spectral_density    import *
from .noise_decomposition import *
from .unit                import *
from .redfield     import *
from .heom         import *

from .noise_decomposition import BathCorrelation
from .solver_base import Result, Integrator

__all__ = [
    # Solvers
    'HEOMSolver',
    'RedfieldSolver',
    # Bath correlation
    'noise_decomposition',
    'BathCorrelation',
    # Spectral densities
    'SpectralDensity',
    'Drude',
    'Brown',
    'OverdampedBrown',
    'BrownDrude',
    # Return types
    'Result',
    'Integrator',
    # Unit system
    'unit',
    # Backend query
    'is_supported',
]

def is_supported(engine):
    engine = engine.lower()
    func = getattr(libheom, f'{engine}_is_supported')
    if func:
        return func()
    else:
        raise Exception(f'Unknown linalg_engine: {engine}')

