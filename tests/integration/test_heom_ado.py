"""Integration test: HEOMSolver across all (space, h_order, liouville_order) combinations.

Verifies that space='ado' (heom_ado) agrees with space='liouville' (heom_liou)
for all combinations of Hilbert-space matrix order and Liouville-space order.

Note on gamma_offdiag: all standard noise models currently store each bath mode
as an independent scalar (1x1 gamma matrix), so gamma_offdiag is always empty
for existing use cases.  The order-dispatch added to heom_ado is a correctness
guarantee for future models with coupled / Jordan-block bath modes.

Reference value: space='liouville', h_order='C', liouville_order='C', n_tiers=3,
rho_00 at t=5.0 (200 steps of dt=2.5e-2, integrator dt=2.5e-3, lsrk4, Eigen).
"""

import numpy as np
import pytest

from pyheom import HEOMSolver, noise_decomposition, Brown

pytestmark = pytest.mark.integration

REFERENCE_RHO00 = 0.7553147376123465
TOL = 1e-13

ALL_COMBINATIONS = [
    pytest.param('liouville', 'C', 'C', id='liou-CC'),
    pytest.param('liouville', 'C', 'F', id='liou-CF'),
    pytest.param('liouville', 'F', 'C', id='liou-FC'),
    pytest.param('liouville', 'F', 'F', id='liou-FF'),
    pytest.param('ado',       'C', 'C', id='ado-CC'),
    pytest.param('ado',       'C', 'F', id='ado-CF'),
    pytest.param('ado',       'F', 'C', id='ado-FC'),
    pytest.param('ado',       'F', 'F', id='ado-FF'),
]


def _run(space, h_order, liouville_order, n_tiers=3, t_final=5.0, callback_dt=0.025):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128, order=h_order)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128, order=h_order)

    qme = HEOMSolver(
        H, [corr],
        space=space, format='dense', engine='eigen',
        liouville_order=liouville_order, solver='lsrk4',
        n_tiers=n_tiers, n_inner_threads=1, n_outer_threads=1,
    )
    rho_0 = np.zeros((2, 2), dtype=np.complex128, order=h_order)
    rho_0[0, 0] = 1.0
    qme.solve(rho_0, np.arange(0.0, t_final + callback_dt * 0.5, callback_dt), dt=2.5e-3)
    return float(qme.rho[0, 0].real)


# ---------------------------------------------------------------------------
# All 8 combinations must match the canonical reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("space,h_order,liouville_order", ALL_COMBINATIONS)
def test_all_combinations_match_reference(space, h_order, liouville_order):
    rho00 = _run(space, h_order, liouville_order)
    assert rho00 == pytest.approx(REFERENCE_RHO00, abs=TOL), \
        f"space={space} h_order={h_order} l_order={liouville_order}: {rho00}"


# ---------------------------------------------------------------------------
# ado must agree with liouville for every (h_order, liouville_order) pair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_order,liouville_order", [
    pytest.param('C', 'C', id='CC'),
    pytest.param('C', 'F', id='CF'),
    pytest.param('F', 'C', id='FC'),
    pytest.param('F', 'F', id='FF'),
])
def test_ado_matches_liouville(h_order, liouville_order):
    rho_liou = _run('liouville', h_order, liouville_order)
    rho_ado  = _run('ado',       h_order, liouville_order)
    assert rho_ado == pytest.approx(rho_liou, abs=TOL), \
        f"h_order={h_order} l_order={liouville_order}: ado={rho_ado} liou={rho_liou}"


# ---------------------------------------------------------------------------
# Trace preservation for ado across all order combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_order,liouville_order", [
    pytest.param('C', 'C', id='CC'),
    pytest.param('C', 'F', id='CF'),
    pytest.param('F', 'C', id='FC'),
    pytest.param('F', 'F', id='FF'),
])
def test_ado_trace_conservation(h_order, liouville_order):
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128, order=h_order)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128, order=h_order)
    qme = HEOMSolver(
        H, [corr],
        space='ado', format='dense', engine='eigen',
        liouville_order=liouville_order, solver='lsrk4',
        n_tiers=3, n_inner_threads=1, n_outer_threads=1,
    )
    rho_0 = np.zeros((2, 2), dtype=np.complex128, order=h_order)
    rho_0[0, 0] = 1.0
    traces = []
    t_list = np.arange(0.0, 5.0 + 0.025 * 0.5, 0.025)

    def callback(t):
        traces.append(float(np.trace(qme.rho).real))

    qme.solve(rho_0, t_list, callback=callback, dt=2.5e-3)
    for i, tr in enumerate(traces):
        assert tr == pytest.approx(1.0, abs=1e-12), \
            f"h={h_order} l={liouville_order}: trace violated at step {i}: {tr}"


# ---------------------------------------------------------------------------
# Consistency across n_tiers for ado vs liouville
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tiers", [1, 2, 3, 4])
@pytest.mark.parametrize("h_order,liouville_order", [
    pytest.param('C', 'C', id='CC'),
    pytest.param('F', 'F', id='FF'),
])
def test_ado_liouville_agreement_tiers(n_tiers, h_order, liouville_order):
    rho_liou = _run('liouville', h_order, liouville_order, n_tiers=n_tiers)
    rho_ado  = _run('ado',       h_order, liouville_order, n_tiers=n_tiers)
    assert rho_ado == pytest.approx(rho_liou, abs=TOL), \
        f"n_tiers={n_tiers} h={h_order} l={liouville_order}: ado={rho_ado} liou={rho_liou}"
