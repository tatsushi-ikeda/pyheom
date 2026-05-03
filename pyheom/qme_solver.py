#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

import numpy as np
import pyheom.pylibheom as libheom
from collections import OrderedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .unit              import *
from .const             import *
from .coo_matrix        import *
from .noise_decomposition import BathCorrelation


@dataclass
class Result:
    """Return value of :meth:`qme_solver.solve`.

    Attributes
    ----------
    times  : ndarray, shape (n_times,)
        Output times (in user units, same as the `t_list` passed to `solve`).
    expect : list of ndarray
        `expect[i]` is a complex array of shape `(n_times,)` containing
        `Tr(e_ops[i] @ rho(t))` at each output time.  Empty list when no
        `e_ops` were given.
    states : object
        Reserved for future use; currently always `None`.
    """
    times:  np.ndarray
    expect: list = field(default_factory=list)
    states: object = None


class Integrator:
    """Low-level integrator interface returned by :meth:`qme_solver.init`.

    Provides step-by-step time evolution with full access to the ADO hierarchy
    after each `advance_to` call.  Intended for nonlinear spectroscopy
    calculations where the hierarchy state is branched at pulse interaction times.

    The integrator shares `_rho` with the parent solver; calling
    `qme.solve()` after `qme.init()` will reset that buffer.
    """

    def __init__(self, qme, dt, **kwargs):
        self._qme    = qme
        self._t      = 0.0
        self._dt     = dt
        self._kwargs = kwargs

    def advance_to(self, t):
        """Advance the state to time *t* (in user units)."""
        delta = t - self._t
        if delta < 0:
            raise ValueError(
                f'Cannot advance backward: current t={self._t}, requested t={t}'
            )
        if delta == 0:
            return
        uf = self._qme._unit_factor
        self._qme.qme_solver_impl.solve(
            self._qme._rho,
            np.array([0.0, delta]) * uf,
            lambda _t: None,
            dt=self._dt * uf,
            **self._kwargs,
        )
        self._t = t

    @property
    def t(self):
        """Current simulation time (user units)."""
        return self._t

    @property
    def rho(self):
        """Physical density matrix (view of the internal buffer)."""
        return self._qme.rho

    @property
    def rho_hierarchy(self):
        """Full ADO hierarchy (view of the internal buffer).

        Shape is `(storage_size, n_level, n_level)`.  Call `.copy()`
        before the next `advance_to` if you need to preserve this state.
        """
        return self._qme._rho


class qme_solver(ABC):
    """Abstract base for HEOM and Redfield solvers.

    engine: 'eigen' (CPU, default), 'mkl' (Intel MKL), or 'cuda' (GPU).
    units: dict `{'energy': unit.X, 'time': unit.Y}`; omit for dimensionless.
    device: CUDA device index -- only valid when engine='cuda'.
    """

    space_char = {
        'hilbert':   'h',
        'liouville': 'l',
    }

    qme_name = 'qme'

    compulsory_args = []
    optional_args   = OrderedDict()

    def get_config(self,
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
        try:
            return getattr(libheom, class_name)
        except AttributeError:
            raise AttributeError(f'unknown libheom class: {class_name}') from None

    def __init__(self,
                 H,
                 noises,
                 space='hilbert',
                 format='dense',
                 engine='eigen',
                 unrolling=True,
                 solver='lsrk4',
                 liouville_order='C',
                 engine_args=None,
                 units=None,
                 device=None,
                 **args):
        engine = engine.lower()
        if engine_args is None:
            engine_args = {}

        # device= is shorthand for engine_args={'device': ...} on CUDA
        if device is not None:
            if engine != 'cuda':
                raise ValueError(
                    f"'device' parameter is only valid for engine='cuda', got '{engine}'"
                )
            engine_args = dict(engine_args, device=device)

        if units is not None:
            self._unit_factor = calc_unit_from_dict(units)
        else:
            self._unit_factor = 1.0

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
                                       liouville_order)

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

        # Resolve callable defaults at construction time (not at module import)
        for k in list(self.qme_args):
            if callable(self.qme_args[k]):
                self.qme_args[k] = self.qme_args[k]()

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
            bc = noises[u]
            if isinstance(bc, BathCorrelation):
                _gamma, _phi_0, _sigma = bc.gamma, bc.phi_0, bc.sigma
                _s_mat, _a_mat         = bc.s_mat, bc.a_mat
                _V, _s_delta           = bc.V, bc.s_delta
            else:
                _gamma, _phi_0, _sigma = bc["gamma"], bc["phi_0"], bc["sigma"]
                _s_mat, _a_mat         = bc["s_mat"], bc["a_mat"]
                _V, _s_delta           = bc["V"], bc["s_delta"]
            gamma   = _gamma.astype(self.dtype, copy=False)
            phi_0   = _phi_0.astype(self.dtype, copy=False)
            sigma   = _sigma.astype(self.dtype, copy=False)
            s_mat   = _s_mat.astype(self.dtype, copy=False)
            a_mat   = _a_mat.astype(self.dtype, copy=False)
            V       = _V.astype(np.complex128, copy=False)
            s_delta = complex(_s_delta)
            self.noises.append(type("noise", (object,),
                                    dict(gamma=gamma,
                                         phi_0=phi_0,
                                         sigma_s=s_mat.T@sigma,
                                         sigma_a=a_mat.T@sigma,
                                         s_delta=s_delta)))
            self.qme_impl.set_noise(u,
                                    libheom_coo_matrix(V),
                                    libheom_coo_matrix(gamma),
                                    phi_0,
                                    sigma,
                                    libheom_coo_matrix(s_mat),
                                    s_delta,
                                    libheom_coo_matrix(a_mat))
        self.qme_impl.set_param(self.engine_impl)

        self.solver_impl     = qme_solver.get_class('{solver}_{c_dtype}{c_order}_{engine}', self.config)()

        self.qme_solver_impl = qme_solver.get_class('qme_solver_{c_dtype}{c_order}_{engine}', self.config)(
            self.engine_impl,
            self.qme_impl,
            self.solver_impl
        )

        # _rho must be C-contiguous: C++ addresses each hierarchy block as a
        # flat array at offset lidx*n_level^2 regardless of the density-matrix
        # order template parameter.  F-order would shift numpy's strides to
        # stride-S intervals, causing reads from the wrong memory positions.
        self._rho = np.zeros((self.storage_size(), self.n_level, self.n_level),
                             dtype=self.dtype, order='C')

    @property
    def rho(self):
        if self.c_contiguous:
            return self._rho[0]
        else:
            return self._rho[0].T  # col_major: _rho stores transposed data; undo for Python

    @property
    def rho_hierarchy(self):
        """Full ADO hierarchy, shape `(storage_size, n_level, n_level)`.

        Returns a view into the internal buffer; the shape may change in a
        future release when adaptive hierarchy filtering is supported.
        """
        return self._rho

    def _init_rho(self, rho_0):
        """Copy *rho_0* into the internal buffer, with shape dispatch."""
        rho_0 = np.asarray(rho_0)
        if rho_0.ndim == 2:
            if rho_0.flags.c_contiguous != self.c_contiguous:
                order_H   = ORDER_CHAR_NUMPY[self.c_contiguous]
                order_rho = ORDER_CHAR_NUMPY[rho_0.flags.c_contiguous]
                raise ValueError(
                    f'H and rho_0 have inconsistent memory order: {order_H} vs {order_rho}'
                )
            if rho_0.dtype != self.dtype:
                raise TypeError(
                    f'H and rho_0 have inconsistent dtype: {self.dtype} vs {rho_0.dtype}'
                )
            self._rho[:] = 0
            if self.c_contiguous:
                self._rho[0] = rho_0
            else:
                self._rho[0] = rho_0.T  # col_major: store transposed so C++ reads col-major layout
        elif rho_0.ndim == 3:
            # 3-D input: restart from a full hierarchy snapshot
            expected = self._rho.shape
            if rho_0.shape != expected:
                raise ValueError(
                    f'Hierarchy rho_0 shape {rho_0.shape} != expected {expected}; '
                    f'use qme.storage_size() to get the correct leading dimension'
                )
            self._rho[:] = rho_0
        else:
            raise ValueError(
                f'rho_0 must be 2-D (n_level, n_level) or 3-D (storage_size, n_level, n_level), '
                f'got shape {rho_0.shape}'
            )

    def solve(self, rho_0, t_list, callback=lambda t: None, e_ops=None, **kwargs):
        """Evolve the system and return a `Result`.

        Parameters
        ----------
        rho_0 : array-like, shape (n_level, n_level) or (storage_size, n_level, n_level)
            Initial density matrix.  The 3-D form restarts from a full hierarchy
            state (e.g. obtained from a previous `rho_hierarchy` snapshot).
        t_list : array-like
            Sequence of output times (user units).
        callback : callable, optional
            Called at each output time with the current time as argument.
        e_ops : list of array-like, optional
            Operators whose expectation values `Tr(O @ rho(t))` are collected
            at every output time.  Results appear in `Result.expect`.

        Returns
        -------
        Result
            Object with attributes `times`, `expect`.
        """
        self._init_rho(rho_0)

        t_arr        = np.asarray(t_list, dtype=float)
        expect_lists = [[] for _ in (e_ops or [])]

        # internal combined callback collects expectation values
        def _cb(t):
            if e_ops is not None:
                rho = self.rho
                for i, op in enumerate(e_ops):
                    expect_lists[i].append(np.trace(op @ rho))
            callback(t)

        if 'dt' in kwargs:
            kwargs = dict(kwargs, dt=kwargs['dt'] * self._unit_factor)
        self.qme_solver_impl.solve(self._rho, t_arr * self._unit_factor, _cb, **kwargs)

        return Result(
            times=t_arr,
            expect=[np.array(lst) for lst in expect_lists],
        )

    def init(self, rho_0, dt, **kwargs):
        """Return a low-level `Integrator` for step-by-step evolution.

        Parameters
        ----------
        rho_0 : array-like, shape (n_level, n_level) or (storage_size, n_level, n_level)
            Initial condition (see `solve` for the 3-D hierarchy form).
        dt : float
            Integration step size (user units).
        **kwargs
            Additional keyword arguments forwarded to the underlying C++ solver
            (e.g. `atol`, `rtol` for `rkdp`).

        Returns
        -------
        Integrator
        """
        self._init_rho(rho_0)
        return Integrator(self, dt=dt, **kwargs)

    @abstractmethod
    def storage_size(self):
        return 1
