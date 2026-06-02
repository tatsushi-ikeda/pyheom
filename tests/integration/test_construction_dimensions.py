"""Regression test for the construction-time segfault at n_level >= 4.

The eigen backend used to segfault during RedfieldSolver / HEOMSolver
construction (set_param) for larger systems in AVX-enabled release builds. Two
alignment bugs were responsible: a `new[]` (aligned) / `delete[]` (non-aligned)
mismatch in the device-buffer helpers, and Eigen `Map<>` wrappers declared with
an alignment the underlying buffers did not satisfy, which made Eigen emit
aligned AVX stores on misaligned memory.

This builds and steps both solvers across n_level x format x liouville_order so
a regression resurfaces here. The DEFINITIVE guard for this class of bug is an
AddressSanitizer (or AVX release) build run -- a plain dev build may not emit the
faulting instruction -- but constructing the full matrix here is the cheap
first line of defense.
"""

import numpy as np
import pytest

from pyheom import RedfieldSolver, HEOMSolver, noise_decomposition, Drude

pytestmark = pytest.mark.integration

# 2/3/4 use static n_level templates, 5/8 the dynamic path -- cross the boundary.
N_LEVELS = [2, 3, 4, 5, 8]
FORMATS = ['dense', 'sparse']
ORDERS = ['C', 'F']


def _model(n):
    H = np.diag(np.arange(n, dtype=float)).astype(np.complex128)
    V = np.ones((n, n), dtype=np.complex128) - np.eye(n)
    corr = noise_decomposition(Drude(0.1, 10.0), T=1.0,
                               type_ltc='psd', n_psd=1, type_psd='n-1/n')
    corr.V = V
    rho_0 = np.zeros((n, n), dtype=np.complex128)
    rho_0[0, 0] = 1.0
    return H, [corr], rho_0


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n", N_LEVELS)
def test_redfield_construct_and_step(n, fmt, order):
    H, corr, rho_0 = _model(n)
    qme = RedfieldSolver(H, corr, space='Liouville', format=fmt, engine='eigen',
                         liouville_order=order, solver='lsrk4')
    qme.solve(rho_0, np.arange(0.0, 3.0, 1.0), dt=0.05)
    # Primary guard: construction + first solve complete without a segfault.
    assert qme.rho.shape == (n, n)
    assert np.all(np.isfinite(qme.rho))


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n", N_LEVELS)
def test_heom_construct_and_step(n, fmt, order):
    H, corr, rho_0 = _model(n)
    qme = HEOMSolver(H, corr, space='ADO', format=fmt, engine='eigen',
                     liouville_order=order, solver='lsrk4', truncation_depth=2)
    qme.solve(rho_0, np.arange(0.0, 3.0, 1.0), dt=0.05)
    # Primary guard: construction + first solve complete without a segfault.
    assert qme.rho.shape == (n, n)
    assert np.all(np.isfinite(qme.rho))
