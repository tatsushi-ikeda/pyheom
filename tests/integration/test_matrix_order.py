"""Integration test: C-contiguous vs F-contiguous and liouville_order='C'/'F'.

All four combinations of (H order) x (liouville_order) must yield identical
population dynamics, verifying that the memory-layout flags are pure
performance settings with no effect on computed values.
"""

import numpy as np
import pytest

from pyheom import heom_solver, noise_decomposition, Brown

pytestmark = pytest.mark.integration

# Reference rho_00 at t~=4.975 from C+C variant (n_tiers=3, lsrk4, dt=2.5e-3)
REFERENCE_RHO00 = 0.7569752673540563
TOL = 1e-12


def _run_one(h_order, liouville_order, n_tiers=3):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)

    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128, order=h_order)
    V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128, order=h_order)
    corr.V = V

    qme = heom_solver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order=liouville_order, solver='lsrk4',
        n_tiers=n_tiers, n_inner_threads=1, n_outer_threads=1,
    )

    rho_0 = np.zeros((2, 2), dtype=np.complex128, order=h_order)
    rho_0[0, 0] = 1.0

    target_t = 4.975
    callback_dt = 2.5e-2
    t_list = np.arange(0.0, target_t + callback_dt * 2, callback_dt)
    captured = {}

    def callback(t):
        if target_t not in captured and abs(t - target_t) < callback_dt * 0.51:
            captured[target_t] = float(qme.rho[0, 0].real)

    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)
    return captured[target_t]


class TestMatrixOrder:

    @pytest.mark.parametrize("h_order,liouville_order", [
        ('C', 'C'),
        ('C', 'F'),
        ('F', 'C'),
        ('F', 'F'),
    ])
    def test_all_orders_match_reference(self, h_order, liouville_order):
        rho00 = _run_one(h_order, liouville_order)
        assert rho00 == pytest.approx(REFERENCE_RHO00, abs=TOL), \
            f"H-order={h_order} L-order={liouville_order}: {rho00} != {REFERENCE_RHO00}"

    def test_c_and_f_h_order_agree(self):
        rho_c = _run_one('C', 'C')
        rho_f = _run_one('F', 'C')
        assert rho_c == pytest.approx(rho_f, abs=TOL)

    def test_c_and_f_liouville_order_agree(self):
        rho_c = _run_one('C', 'C')
        rho_f = _run_one('C', 'F')
        assert rho_c == pytest.approx(rho_f, abs=TOL)
