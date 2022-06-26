#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

import pyheom.pylibheom as libheom

import numpy as np
import pyheom.pylibheom as libheom
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from .unit       import *
from .const      import *
from .coo_matrix import *

class qme_solver:
    __metaclass__ = ABCMeta
    
    space_char = {
        'hilbert':   'h',
        'liouville': 'l',
    }

    qme_name = 'qme'

    compulsory_args = []
    optional_args   = OrderedDict()

    def get_config(self.
                   qme_name,
                   engine,
                   solver,
                   dtype,
                   space,
                   format,
                   c_contiguous,
                   n_level,
                   unrolling,
                   c_contiguous_liouville):
        c_dtype  = DTYPE_CHAR[dtype]
        c_format = FORMAT_CHAR[format.lower()]
        c_order  = ORDER_CHAR[c_contiguous]
        c_space  = self.space_char[space.lower()]
        if unrolling and engine.lower() == 'eigen' and n_level in [2,]:
            c_level = f'{n_level}'
        else:
            c_level = 'n'
        if c_space in ['l', 'a']:
            c_order_liouville = ORDER_CHAR[c_contiguous_liouville]
        else:
            c_order_liouville = ''
        return dict(qme_name=qme_name, engine=engine, solver=solver, c_dtype=c_dtype, c_space=c_space, c_format=c_format, c_order=c_order, c_level=c_level, c_order_liouville=c_order_liouville)

    @staticmethod
    def get_class(name_format, config):
        class_name = name_format.format(**config)
        class_obj  = getattr(libheom, class_name)
        if class_obj:
            return class_obj
        else:
            raise Exception(f'[Error:] unknown class: {class_name}')

    def __init__(self,
                 H,
                 noises,
                 space='hilbert',
                 format='dense',
                 engine='eigen',
                 unrolling=True,
                 solver='lsrk4',
                 order_liouville='row_major',
                 engine_args={},
                 **args):
        engine = engine.lower()

        self.H            = H
        self.noises       = noises
        self.dtype        = H.dtype
        self.n_level      = H.shape[0]
        self.c_contiguous = H.flags.c_contiguous
        self.config  = self.get_config(self.qme_name,
                                       engine,
                                       solver,
                                       self.dtype,
                                       space,
                                       format,
                                       self.c_contiguous,
                                       self.n_level,
                                       unrolling,
                                       order_liouville)

        self.engine_impl = qme_solver.get_class('{engine}', self.config)(
            *(dict(ENGINE_ARGS[engine], **engine_args).values())
        )
        
        given_keys      = set(args.keys())
        compulsory_keys = set(self.compulsory_args)
        optional_keys   = set(self.optional_args.keys())
        if not compulsory_keys.issubset(given_keys):
            missing_keys = compulsory_keys - given_keys
            raise KeyError(f'Missing keyword arguments: {", ".join(missing_keys)}')
        given_keys -= compulsory_keys
        if not given_keys.issubset(optional_keys):
            unknown_keys = given_keys - optional_keys
            raise KeyError(f'Unknown keyword arguments: {", ".join(unknown_keys)}')
        self.qme_args = OrderedDict((key,None) for key in self.compulsory_args)
        self.qme_args.update(self.optional_args)
        self.qme_args.update(args)
        
        self.qme_impl = qme_solver.get_class(
            '{qme_name}_{c_dtype}{c_space}{c_format}{c_order}{c_order_liouville}{c_level}_{engine}',
            self.config)(
            *(self.qme_args.values())
        )
        self.qme_impl.set_system(libheom_coo_matrix(H))

        n_noise = len(noises)
        self.qme_impl.alloc_noises(n_noise)
        self.noises = []
        for u in range(n_noise):
            gamma   = noises[u]["gamma"].astype(self.dtype)
            phi_0   = noises[u]["phi_0"].astype(self.dtype)
            sigma   = noises[u]["sigma"].astype(self.dtype)
            s       = noises[u]["S"].astype(self.dtype)
            a       = noises[u]["A"].astype(self.dtype)
            S_delta = complex(noises[u]["s_delta"])
            self.noises.append(type("noise", (object,),
                                    dict(gamma=gamma,
                                         phi_0=phi_0,
                                         sigma_s=s.T@sigma,
                                         sigma_a=a.T@sigma,
                                         S_delta=S_delta)))
            self.qme_impl.set_noise(u,
                                    libheom_coo_matrix(noises[u]["V"].astype(np.complex128)),
                                    libheom_coo_matrix(gamma),
                                    phi_0,
                                    sigma,
                                    libheom_coo_matrix(s),
                                    S_delta,
                                    libheom_coo_matrix(a))
        self.qme_impl.set_param(self.engine_impl)
        
        self.solver_impl     = qme_solver.get_class('{solver}_{c_dtype}{c_order}_{engine}', self.config)()
        
        self.qme_solver_impl = qme_solver.get_class('qme_solver_{c_dtype}{c_order}_{engine}', self.config)(
            self.engine_impl,
            self.qme_impl,
            self.solver_impl
        )
    
    def solve(self, rho, t_list, callback=lambda t: None, **kwargs):
        if rho.flags.c_contiguous != self.c_contiguous:
            order_H   = ORDER_CHAR_NUMPY[self.c_contiguous]
            order_rho = ORDER_CHAR_NUMPY[rho.flags.c_contiguous]
            raise ValueError(f'The orders of H and rho are inconsistent: {order_H} and {order_rho}')
        if rho.dtype != self.dtype:
            raise TypeError(f'The types of H and rho are inconsistent: {self.dtype} and {rho.dtype}')
        if 'dt' in kwargs:
            kwargs = dict(kwargs, dt=kwargs['dt']*calc_unit())
        self.qme_solver_impl.solve(rho, np.array(t_list)*calc_unit(), callback, **kwargs)

    @abstractmethod
    def storage_size(self):
        return 1

