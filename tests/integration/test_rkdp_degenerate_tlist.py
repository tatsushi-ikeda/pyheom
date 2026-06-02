"""Regression: rkdp must not hang on a degenerate (zero-span) t_list interval.

A zero-span interval (t_list[i+1] == t_list[i]) used to make the adaptive solver
clamp its step size to 0 and persist that 0, so a later nonzero solve on the same
instance looped forever (CPU at 99%). solve() now skips zero-span intervals (the
callback still fires; rho is unchanged at the repeated time), so all solvers
behave the same. A decreasing t_list is rejected with ValueError.

The hang is an infinite loop, so the reproduction runs in a child process guarded
by a join timeout: a regression shows up as a timeout (test failure), not a hung
suite.
"""

import multiprocessing as mp

import numpy as np
import pytest

from pyheom import HEOMSolver, noise_decomposition, Brown

pytestmark = pytest.mark.integration

HANG_TIMEOUT_S = 60


def _build():
    J = Brown(0.01, 0.5, 1.0)
    corr = noise_decomposition(J, T=1.0, type_ltc='psd', n_psd=1, type_psd='n-1/n')
    omega_1 = np.sqrt(1.0 - 0.5**2 * 0.25)
    H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
    corr.V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return HEOMSolver(
        H, [corr],
        space='liouville', format='dense', engine='eigen',
        liouville_order='C', solver='rkdp',
        truncation_depth=3, n_inner_threads=1, n_outer_threads=1,
    )


def _rho0():
    r = np.zeros((2, 2), dtype=np.complex128)
    r[0, 0] = 1.0
    return r


def _worker(q):
    kw = dict(dt=0.25e-2, atol=1e-8, rtol=1e-6)
    try:
        # (a) single solve with a duplicate interior time vs the same grid without it.
        qme = _build()
        qme.solve(_rho0(), [0.0, 0.05, 0.05, 0.10], **kw)
        rho_dup = np.asarray(qme.rho)

        qme_ref = _build()
        qme_ref.solve(_rho0(), [0.0, 0.05, 0.10], **kw)
        rho_ref = np.asarray(qme_ref.rho)

        # (b) the reported scenario: separate calls on ONE instance,
        #     nonzero -> zero-span [t, t] -> nonzero. The middle call used to
        #     poison the persisted step size, hanging the third call.
        qme2 = _build()
        qme2.solve(_rho0(), [0.0, 0.05], **kw)
        qme2.solve(_rho0(), [0.0, 0.0], **kw)
        qme2.solve(_rho0(), [0.0, 0.05], **kw)
        rho_seq = np.asarray(qme2.rho)

        q.put(('ok', rho_dup, rho_ref, rho_seq))
    except BaseException as exc:  # surface any error to the parent
        q.put(('err', repr(exc)))


def test_rkdp_degenerate_tlist_does_not_hang():
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q,))
    p.start()
    p.join(HANG_TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        pytest.fail(
            'rkdp solve() hung on a degenerate (zero-span) t_list interval '
            f'(no result within {HANG_TIMEOUT_S}s)'
        )

    status, *payload = q.get_nowait()
    assert status == 'ok', f'worker raised: {payload[0]}'
    rho_dup, rho_ref, rho_seq = payload
    assert np.all(np.isfinite(rho_dup))
    assert np.all(np.isfinite(rho_seq))
    # A duplicate output time must be a no-op: same result as the grid without it.
    np.testing.assert_allclose(rho_dup, rho_ref, atol=1e-10)


def test_decreasing_tlist_raises():
    qme = _build()
    with pytest.raises(ValueError):
        qme.solve(_rho0(), [0.0, 0.10, 0.05], dt=0.25e-2, atol=1e-8, rtol=1e-6)
