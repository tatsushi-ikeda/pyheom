"""Integration test: heom_solver with space='ado' (heom_ado path).

Verifies that the ADO representation (space='ado') gives results numerically
identical to the on-the-fly Liouville-space propagation (space='liouville').
Serves as a regression test for the try-catch -> find() refactoring in
heom_ado::set_param (M2).
"""

import copy
import numpy as np
import pytest

from pyheom import heom_solver, noise_decomposition, Brown

pytestmark = pytest.mark.integration


def _corr_and_H():
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    return corr, H


def _run(space, corr, H, n_tiers=3, t_final=5.0, dt=2.5e-3):
    c = copy.deepcopy(corr)
    c.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    qme = heom_solver(
        H, [c],
        space=space, format='dense', engine='eigen',
        liouville_order='C', solver='lsrk4',
        n_tiers=n_tiers, n_inner_threads=1, n_outer_threads=1,
    )
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0
    qme.solve(rho_0, np.arange(0.0, t_final + dt * 0.5, dt * 10), dt=dt)
    return qme.rho.copy()


class TestHeomAdo:

    def test_ado_matches_liouville(self):
        """space='ado' and space='liouville' must agree to machine precision."""
        corr, H = _corr_and_H()
        rho_liou = _run('liouville', corr, H)
        rho_ado  = _run('ado',       corr, H)
        np.testing.assert_allclose(rho_ado, rho_liou, atol=1e-14,
                                   err_msg="ado and liouville results differ")

    def test_ado_trace_conservation(self):
        """Trace of rho must equal 1 throughout the ADO evolution."""
        corr, H = _corr_and_H()
        c = copy.deepcopy(corr)
        c.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        qme = heom_solver(
            H, [c],
            space='ado', format='dense', engine='eigen',
            liouville_order='C', solver='lsrk4',
            n_tiers=3, n_inner_threads=1, n_outer_threads=1,
        )
        rho_0 = np.zeros((2, 2), dtype=np.complex128)
        rho_0[0, 0] = 1.0

        traces = []
        t_list = np.arange(0.0, 5.0 + 2.5e-2 * 0.5, 2.5e-2)

        def callback(t):
            traces.append(float(np.trace(qme.rho).real))

        qme.solve(rho_0, t_list, callback=callback, dt=2.5e-3)

        for i, tr in enumerate(traces):
            assert tr == pytest.approx(1.0, abs=1e-12), \
                f"trace violated at step {i}: {tr}"

    @pytest.mark.parametrize("n_tiers", [1, 2, 3, 4])
    def test_ado_liouville_agreement_tiers(self, n_tiers):
        """Agreement must hold for multiple truncation depths."""
        corr, H = _corr_and_H()
        rho_liou = _run('liouville', corr, H, n_tiers=n_tiers)
        rho_ado  = _run('ado',       corr, H, n_tiers=n_tiers)
        np.testing.assert_allclose(rho_ado, rho_liou, atol=1e-14,
                                   err_msg=f"n_tiers={n_tiers}: ado vs liouville differ")
