"""Integration test: Eigen, MKL, and CUDA backends agree to 1e-8.

Each non-Eigen backend is compared to the Eigen reference for the same
system (2-level Brownian oscillator, n_tiers=3, lsrk4). Tests are skipped
when the corresponding backend is not available at runtime.
"""

import numpy as np
import pytest

import pyheom.pylibheom as _lb
from pyheom import HEOMSolver, noise_decomposition, Brown, unit

pytestmark = pytest.mark.integration

TARGET_T = 4.975
CALLBACK_DT = 2.5e-2

skip_if_no_mkl  = pytest.mark.skipif(not _lb.mkl_is_supported(),  reason="MKL backend not available")
skip_if_no_cuda = pytest.mark.skipif(not _lb.cuda_is_supported(), reason="CUDA backend not available")


def _run(engine):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    qme = HEOMSolver(
        H, [corr],
        space='liouville', format='dense', engine=engine,
        liouville_order='C', solver='lsrk4',
        n_tiers=3, n_inner_threads=1, n_outer_threads=1,
    )

    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0
    t_list = np.arange(0.0, TARGET_T + CALLBACK_DT * 2, CALLBACK_DT)
    captured = {}

    def callback(t):
        if TARGET_T not in captured and abs(t - TARGET_T) < CALLBACK_DT * 0.51:
            captured[TARGET_T] = float(qme.rho[0, 0].real)

    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)
    return captured[TARGET_T]


class TestBackendAgreement:

    @skip_if_no_mkl
    def test_mkl_agrees_with_eigen(self):
        rho_eigen = _run('eigen')
        rho_mkl   = _run('mkl')
        assert rho_mkl == pytest.approx(rho_eigen, abs=1e-8)

    @skip_if_no_cuda
    def test_cuda_agrees_with_eigen(self):
        rho_eigen = _run('eigen')
        rho_cuda  = _run('cuda')
        assert rho_cuda == pytest.approx(rho_eigen, abs=1e-8)

    @pytest.mark.skipif(
        not (_lb.mkl_is_supported() and _lb.cuda_is_supported()),
        reason="Both MKL and CUDA backends required",
    )
    def test_mkl_cuda_agree(self):
        rho_mkl  = _run('mkl')
        rho_cuda = _run('cuda')
        assert rho_mkl == pytest.approx(rho_cuda, abs=1e-8)
