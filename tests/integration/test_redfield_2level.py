"""Integration test: 2-level Brownian-oscillator Redfield dynamics.

Same physical system as test_brownian_2level.py but using RedfieldSolver
(Markov/secular approximation).  Reference values differ from HEOM because
Redfield uses a weaker approximation: no hierarchy, secular/Markov.
"""

import numpy as np
import pytest

from pyheom import RedfieldSolver, noise_decomposition, Brown, unit


# Pinned reference: rho_00(t) from Eigen build, lsrk4, dt=2.5e-3
REFERENCE = [
    ( 0.000, 1.0),
    ( 4.975, 0.5700785921943186),
    ( 9.975, 0.3946379785374777),
    (14.975, 0.3235874426885779),
    (24.975, 0.2831600465616965),
]
TOL = 1e-10


pytestmark = pytest.mark.integration


def _build_solver():
    lambda_0 = 0.01
    omega_0  = 1.0
    zeta     = 0.5
    T        = 1.0

    J = Brown(lambda_0, zeta, omega_0)
    corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')

    omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    return RedfieldSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver='lsrk4',
    )


def test_redfield_2level_population_dynamics():
    qme = _build_solver()
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0

    target_times = [t for t, _ in REFERENCE if t > 0.0]
    callback_dt  = 2.5e-2
    t_list = np.arange(0.0, target_times[-1] + callback_dt * 0.5, callback_dt)

    captured = {}

    def callback(t):
        for ref_t in target_times:
            if ref_t in captured:
                continue
            if abs(t - ref_t) < callback_dt * 0.51:
                captured[ref_t] = float(qme.rho[0, 0].real)

    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)

    for ref_t, ref_val in REFERENCE[1:]:
        assert ref_t in captured, f"callback missed t={ref_t}"
        assert captured[ref_t] == pytest.approx(ref_val, abs=TOL), \
            f"t={ref_t}: got {captured[ref_t]}, expected {ref_val}"


def test_redfield_population_conservation():
    """Trace of rho must be 1 throughout the evolution."""
    qme = _build_solver()
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0

    traces = []
    callback_dt = 2.5e-2
    t_list = np.arange(0.0, 5.0 + callback_dt * 0.5, callback_dt)

    def callback(t):
        traces.append(float((qme.rho[0, 0] + qme.rho[1, 1]).real))

    qme.solve(rho_0, t_list, callback=callback, dt=0.25e-2)

    for i, tr in enumerate(traces):
        assert tr == pytest.approx(1.0, abs=1e-12), \
            f"trace violated at step {i}: {tr}"


def test_redfield_equilibrium_approach():
    """At long times rho_00 approaches the thermal equilibrium value."""
    qme = _build_solver()
    rho_0 = np.zeros((2, 2), dtype=np.complex128)
    rho_0[0, 0] = 1.0

    t_list = np.arange(0.0, 25.0 + 2.5e-2 * 0.5, 2.5e-2)
    qme.solve(rho_0, t_list, dt=0.25e-2)
    final_pop = float(qme.rho[0, 0].real)

    # thermal: p_0 = 1/(1+exp(omega_1/T)), omega_1 = sqrt(1-0.0625)
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    p_eq = 1.0 / (1.0 + np.exp(omega_1))  # at T=1

    assert final_pop == pytest.approx(p_eq, abs=0.02), \
        f"final population {final_pop:.4f} not near equilibrium {p_eq:.4f}"
