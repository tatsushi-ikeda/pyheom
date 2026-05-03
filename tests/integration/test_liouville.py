"""Integration test: Liouvillian properties.

At zero bath coupling the dynamics reduces to free Hamiltonian evolution
  rho(t) = exp(-iHt) rho_0 exp(iHt).

Also verifies: dimension consistency for n_level=2 and 3; trace preservation.
"""

import numpy as np
import pytest

from pyheom import redfield_solver, noise_decomposition, Drude, unit

pytestmark = pytest.mark.integration


def _free_evolution(H, rho_0, t):
    """Exact free-Hamiltonian evolution: exp(-iHt) rho_0 exp(+iHt)."""
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * t)
    U = evecs * phases                # columns: evecs, each scaled by phase
    return U @ (evecs.conj().T @ rho_0 @ evecs) @ U.conj().T


def _build_solver(H, V, eta):
    """Redfield solver with coupling strength eta (Drude bath)."""
    J = Drude(eta=eta, gamma_c=1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='none')
    corr.V = V
    return redfield_solver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver='lsrk4',
    )


# ---------------------------------------------------------------------------
# Zero-coupling limit: 2-level system
# ---------------------------------------------------------------------------

class TestFreeEvolution2Level:
    """With eta->0, solver output matches free-Hamiltonian evolution."""

    def setup_method(self):
        self.omega = 1.0
        self.H = np.array([[self.omega, 0.0], [0.0, 0.0]], dtype=np.complex128)
        self.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        # coherent initial state
        self.rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

    def _run(self, eta, t_final):
        qme = _build_solver(self.H, self.V, eta)
        t_list = np.linspace(0.0, t_final, 21)
        qme.solve(self.rho_0, t_list, dt=1e-4)
        return qme.rho.copy()

    @pytest.mark.parametrize("t_final", [0.5, 1.0])
    def test_small_coupling_approx_free_evolution(self, t_final):
        eta = 1e-8
        rho_qme  = self._run(eta, t_final)
        rho_free = _free_evolution(self.H, self.rho_0, t_final)
        # error should be O(eta * t)
        assert np.max(np.abs(rho_qme - rho_free)) < 1e-7

    def test_error_scales_linearly_with_coupling(self):
        t = 1.0
        rho_free = _free_evolution(self.H, self.rho_0, t)
        err_small = np.max(np.abs(self._run(1e-8, t) - rho_free))
        err_large = np.max(np.abs(self._run(1e-4, t) - rho_free))
        # err_large / err_small ~= 1e4 (within an order of magnitude)
        ratio = err_large / err_small
        assert 1e3 < ratio < 1e5


# ---------------------------------------------------------------------------
# Dimension consistency: n_level = 2 and 3
# ---------------------------------------------------------------------------

class TestDimensions:

    @pytest.mark.parametrize("n_level", [2, 3])
    def test_rho_shape(self, n_level):
        H = np.diag(np.arange(n_level, dtype=np.float64)).astype(np.complex128)
        V = (np.ones((n_level, n_level)) - np.eye(n_level)).astype(np.complex128)
        J = Drude(eta=1e-8, gamma_c=1.0)
        corr = noise_decomposition(J, T=1.0, type_ltc='none')
        corr.V = V
        qme = redfield_solver(H, [corr], space='liouville',
                              format='dense', engine='eigen',
                              liouville_order='C', solver='lsrk4')
        rho_0 = np.eye(n_level, dtype=np.complex128) / n_level
        qme.solve(rho_0, np.array([0.0, 0.5, 1.0]), dt=1e-4)
        assert qme.rho.shape == (n_level, n_level)


# ---------------------------------------------------------------------------
# Trace preservation
# ---------------------------------------------------------------------------

class TestTracePreservation:

    def test_trace_preserved_short_time(self):
        H = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        J = Drude(eta=0.1, gamma_c=1.0)
        corr = noise_decomposition(J, T=1.0, type_ltc='none')
        corr.V = V
        qme = redfield_solver(H, [corr], space='liouville',
                              format='dense', engine='eigen',
                              liouville_order='C', solver='lsrk4')

        rho_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        traces = []
        t_list = np.linspace(0.0, 5.0, 51)

        def callback(t):
            traces.append(float(np.trace(qme.rho).real))

        qme.solve(rho_0, t_list, callback=callback, dt=1e-3)

        for i, tr in enumerate(traces):
            assert tr == pytest.approx(1.0, abs=1e-10), \
                f"trace violated at step {i}: {tr}"
