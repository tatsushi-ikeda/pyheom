"""Integration test: numerical consistency under multi-threading.

n_outer_threads > 1 (OMP parallel loop over hierarchy nodes in heom_hilb/heom_liou)
and n_inner_threads > 1 (Eigen::setNbThreads for inner BLAS/Eigen operations) must
both produce results that agree with the single-thread baseline to machine precision.

The OMP loop in heom_liou/heom_hilb writes to disjoint drho_dt[lidx*n_level_2]
memory regions via per-thread temp buffers.  There are no shared writes and no
floating-point reordering across outer threads, so n_outer results are expected
to be bit-for-bit identical to the single-thread baseline (TOL = 0.0 would also
pass; we use 1e-14 as a safety margin against platform differences).

n_inner_threads controls Eigen::setNbThreads which may reorder floating-point
operations within a single gemv/gemm call, so TOL = 1e-14 is used there too.

ADO space has no OMP outer loop (it uses a single global gemv on the full R matrix),
so n_outer is silently accepted but has no effect; the test still checks results agree.
"""
from multiprocessing import cpu_count

import numpy as np
import pytest

import pyheom.pylibheom as _lb
from pyheom import HEOMSolver, noise_decomposition, Brown

pytestmark = pytest.mark.integration

_NCPU = cpu_count()
skip_if_no_cuda     = pytest.mark.skipif(not _lb.cuda_is_supported(),
                                         reason="CUDA backend not available")
skip_if_single_core = pytest.mark.skipif(_NCPU <= 1,
                                         reason="multi-core machine required")

TARGET_T    = 4.975
CALLBACK_DT = 2.5e-2
N_TIERS     = 3
DT          = 2.5e-3
TOL         = 1e-14


def _run(space, n_outer=1, n_inner=1):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    qme = HEOMSolver(
        H, [corr],
        space=space, format='dense', engine='eigen',
        solver='lsrk4',
        truncation_depth=N_TIERS, n_inner_threads=n_inner, n_outer_threads=n_outer,
    )
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0
    t_list = np.arange(0.0, TARGET_T + CALLBACK_DT * 2, CALLBACK_DT)
    captured = {}

    def callback(t):
        if TARGET_T not in captured and abs(t - TARGET_T) < CALLBACK_DT * 0.51:
            captured[TARGET_T] = float(qme.rho[0, 0].real)

    qme.solve(rho_0, t_list, callback=callback, dt=DT)
    return captured[TARGET_T]


# ---------------------------------------------------------------------------
# n_outer_threads
# ---------------------------------------------------------------------------

class TestNOuterThreads:

    @skip_if_single_core
    def test_liouville_dense(self):
        """n_outer > 1 in the heom_liou OMP loop must match n_outer=1."""
        ref  = _run('liouville', n_outer=1)
        para = _run('liouville', n_outer=_NCPU)
        assert para == pytest.approx(ref, abs=TOL)

    @skip_if_single_core
    def test_hilbert_dense(self):
        """n_outer > 1 in the heom_hilb OMP loop must match n_outer=1."""
        ref  = _run('hilbert', n_outer=1)
        para = _run('hilbert', n_outer=_NCPU)
        assert para == pytest.approx(ref, abs=TOL)

    @skip_if_single_core
    def test_ado_ignores_n_outer(self):
        """ADO space has no OMP outer loop; n_outer > 1 must not change results."""
        ref  = _run('ado', n_outer=1)
        para = _run('ado', n_outer=_NCPU)
        assert para == pytest.approx(ref, abs=TOL)


# ---------------------------------------------------------------------------
# n_inner_threads
# ---------------------------------------------------------------------------

class TestNInnerThreads:

    @skip_if_single_core
    def test_liouville_dense(self):
        """n_inner > 1 (Eigen::setNbThreads) must match n_inner=1."""
        ref  = _run('liouville', n_inner=1)
        para = _run('liouville', n_inner=_NCPU)
        assert para == pytest.approx(ref, abs=TOL)

    @skip_if_single_core
    def test_hilbert_dense(self):
        """n_inner > 1 with Hilbert space must match n_inner=1."""
        ref  = _run('hilbert', n_inner=1)
        para = _run('hilbert', n_inner=_NCPU)
        assert para == pytest.approx(ref, abs=TOL)

    @skip_if_single_core
    def test_ado(self):
        """n_inner > 1 with ADO space must match n_inner=1."""
        ref  = _run('ado', n_inner=1)
        para = _run('ado', n_inner=_NCPU)
        assert para == pytest.approx(ref, abs=TOL)


# ---------------------------------------------------------------------------
# CUDA: n_outer_threads > 1 must raise ValueError at construction time
# ---------------------------------------------------------------------------

class TestCudaValidation:

    @skip_if_no_cuda
    def test_cuda_n_outer_gt_1_raises(self):
        """CUDA engine has no OMP outer loop; n_outer > 1 must raise ValueError."""
        J = Brown(0.01, 0.5, 1.0)
        corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
        omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
        H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
        corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        with pytest.raises(ValueError, match='n_outer_threads'):
            HEOMSolver(
                H, [corr],
                space='liouville', format='dense', engine='cuda',
                solver='lsrk4', truncation_depth=N_TIERS,
                n_outer_threads=2,
            )
