#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

from .predefined_noise    import *
from .noise_decomposition import *
from .unit                import *
from .redfield_solver     import *
from .heom_solver         import *

def is_supported(engine):
    engine = engine.lower()
    func = getattr(libheom, f'{engine}_is_supported')
    if func:
        return func()
    else:
        raise Exception(f'Unknown linalg_engine: {engine}')

