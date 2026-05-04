"""Tests for benchmarks._auto: warmup, _tune_trial, tune_threads, auto_select.

Split into two layers:
  - Logic tests: fast, use monkeypatching to avoid running real solvers.
    These verify control flow, parameter forwarding, and result structure.
  - Integration tests: run the actual HEOMSolver; slow but catch runtime bugs.

The benchmark conftest.py skips this entire file when pylibheom is missing.
"""

from multiprocessing import cpu_count
from unittest.mock import MagicMock

import numpy as np
import pytest

import benchmarks._auto as _m  # import module object for monkeypatching
from ._auto import (
    measure_rss_delta, warmup, _tune_trial,
    tune_threads, auto_select,
    _N_TUNE_STEPS, _WARMUP_STEPS, _WARMUP_DT_CALLBACK,
)
from ._core import DT, T_FINAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_qme(n_outer=1, n_inner=1):
    """Return a MagicMock qme carrying thread-count attributes."""
    q = MagicMock()
    q._n_outer = n_outer
    q._n_inner = n_inner
    return q


def _patch_build_solver(monkeypatch, timing_map=None):
    """Patch build_solver to return fake qme objects; optionally controlled."""
    def _build(engine, space, fmt, solver='lsrk4', unrolling=True,
                n_outer_threads=1, n_inner_threads=1):
        return _fake_qme(n_outer_threads, n_inner_threads)
    monkeypatch.setattr(_m, 'build_solver', _build)


def _patch_run_trial(monkeypatch, timing_fn=None):
    """Patch run_trial so no real solver runs.  timing_fn(qme) -> seconds."""
    if timing_fn is None:
        def timing_fn(qme, **kw): return 0.05
    def _run(qme, t_final=None, dt_callback=None):
        return timing_fn(qme)
    monkeypatch.setattr(_m, 'run_trial', _run)


# ---------------------------------------------------------------------------
# measure_rss_delta
# ---------------------------------------------------------------------------

class TestMeasureRssDelta:

    def test_returns_nonnegative(self):
        assert measure_rss_delta(lambda: None) >= 0

    def test_calls_fn_exactly_once(self):
        calls = []
        measure_rss_delta(lambda: calls.append(1))
        assert calls == [1]

    def test_passes_args_to_fn(self):
        received = []
        measure_rss_delta(lambda a, b: received.append((a, b)), 7, 9)
        assert received == [(7, 9)]

    def test_passes_kwargs_to_fn(self):
        received = []
        measure_rss_delta(lambda x=None: received.append(x), x=42)
        assert received == [42]


# ---------------------------------------------------------------------------
# _tune_trial: step count forwarding
# ---------------------------------------------------------------------------

class TestTuneTrial:

    def _capture_run_trial(self, monkeypatch):
        calls = []
        def mock_run_trial(qme, t_final=None, dt_callback=None):
            calls.append({'t_final': t_final, 'dt_callback': dt_callback})
            return 0.01
        monkeypatch.setattr(_m, 'run_trial', mock_run_trial)
        return calls

    def test_default_uses_N_TUNE_STEPS(self, monkeypatch):
        calls = self._capture_run_trial(monkeypatch)
        _tune_trial(_fake_qme())
        assert len(calls) == 1
        assert abs(calls[0]['t_final'] - _N_TUNE_STEPS * DT) < 1e-12

    def test_dt_callback_equals_DT(self, monkeypatch):
        calls = self._capture_run_trial(monkeypatch)
        _tune_trial(_fake_qme())
        assert abs(calls[0]['dt_callback'] - DT) < 1e-12

    @pytest.mark.parametrize('n_steps', [5, 10, 20, 50])
    def test_n_steps_sets_t_final(self, monkeypatch, n_steps):
        calls = self._capture_run_trial(monkeypatch)
        _tune_trial(_fake_qme(), n_steps=n_steps)
        assert abs(calls[0]['t_final'] - n_steps * DT) < 1e-12

    def test_returns_run_trial_value(self, monkeypatch):
        monkeypatch.setattr(_m, 'run_trial', lambda q, **kw: 0.123)
        result = _tune_trial(_fake_qme())
        assert abs(result - 0.123) < 1e-12


# ---------------------------------------------------------------------------
# warmup: parameter forwarding
# ---------------------------------------------------------------------------

class TestWarmup:

    def test_calls_run_trial_with_warmup_t_final(self, monkeypatch):
        calls = []
        monkeypatch.setattr(_m, 'run_trial',
                            lambda q, t_final=None, dt_callback=None:
                            calls.append(t_final) or 0.1)
        warmup(_fake_qme())
        expected = _WARMUP_STEPS * _WARMUP_DT_CALLBACK
        assert abs(calls[0] - expected) < 1e-12

    def test_calls_run_trial_with_warmup_dt_callback(self, monkeypatch):
        calls = []
        monkeypatch.setattr(_m, 'run_trial',
                            lambda q, t_final=None, dt_callback=None:
                            calls.append(dt_callback) or 0.1)
        warmup(_fake_qme())
        assert abs(calls[0] - _WARMUP_DT_CALLBACK) < 1e-12

    def test_returns_elapsed(self, monkeypatch):
        monkeypatch.setattr(_m, 'run_trial', lambda q, **kw: 0.777)
        assert abs(warmup(_fake_qme()) - 0.777) < 1e-12


# ---------------------------------------------------------------------------
# tune_threads: control flow and selection logic
# ---------------------------------------------------------------------------

class TestTuneThreadsLogic:

    def _mock_env(self, monkeypatch, timing_fn=None):
        """Set up fully mocked build_solver, warmup, _tune_trial."""
        _patch_build_solver(monkeypatch)
        monkeypatch.setattr(_m, 'warmup', lambda q: 0.001)
        if timing_fn is None:
            monkeypatch.setattr(_m, '_tune_trial',
                                lambda q, n_steps=_N_TUNE_STEPS: 0.05)
        else:
            monkeypatch.setattr(
                _m, '_tune_trial',
                lambda q, n_steps=_N_TUNE_STEPS: timing_fn(q._n_outer, q._n_inner),
            )

    def test_returns_three_tuple(self, monkeypatch):
        self._mock_env(monkeypatch)
        result = tune_threads('Eigen', 'Hilbert', 'dense')
        assert len(result) == 3

    def test_best_outer_positive(self, monkeypatch):
        self._mock_env(monkeypatch)
        best_outer, best_inner, _ = tune_threads('Eigen', 'Hilbert', 'dense')
        assert best_outer >= 1

    def test_best_inner_positive(self, monkeypatch):
        self._mock_env(monkeypatch)
        _, best_inner, _ = tune_threads('Eigen', 'Hilbert', 'dense')
        assert best_inner >= 1

    def test_best_time_nonnegative(self, monkeypatch):
        self._mock_env(monkeypatch)
        _, _, best_t = tune_threads('Eigen', 'Hilbert', 'dense')
        assert best_t >= 0

    def test_tests_multiple_pairs_on_multicore(self, monkeypatch):
        """Must probe more than (1,1) when cpu_count() > 1."""
        tested = []
        def mock_build(engine, space, fmt, solver='lsrk4', unrolling=True,
                       n_outer_threads=1, n_inner_threads=1):
            tested.append((n_outer_threads, n_inner_threads))
            return _fake_qme(n_outer_threads, n_inner_threads)
        monkeypatch.setattr(_m, 'build_solver', mock_build)
        monkeypatch.setattr(_m, 'warmup', lambda q: 0.001)
        monkeypatch.setattr(_m, '_tune_trial',
                            lambda q, n_steps=_N_TUNE_STEPS: 0.05)

        tune_threads('Eigen', 'Hilbert', 'dense')

        if cpu_count() > 1:
            assert len(tested) > 1, (
                'tune_threads must test multiple (n_outer, n_inner) pairs; '
                'only (1,1) was tested.  This is the OMP_NUM_THREADS regression.'
            )
            assert any(no > 1 or ni > 1 for no, ni in tested)

    def test_selects_omp_dominant_winner(self, monkeypatch):
        """When more outer threads are faster, (cpu_count(), 1) should win."""
        max_t = cpu_count()
        self._mock_env(monkeypatch, timing_fn=lambda no, ni: 1.0 / no)
        best_outer, best_inner, _ = tune_threads('Eigen', 'Hilbert', 'dense')
        assert best_outer == max_t
        assert best_inner == 1

    def test_selects_inner_dominant_winner(self, monkeypatch):
        """When more inner threads are faster, (1, cpu_count()) should win."""
        max_t = cpu_count()
        self._mock_env(monkeypatch, timing_fn=lambda no, ni: 1.0 / ni)
        best_outer, best_inner, _ = tune_threads('Eigen', 'Hilbert', 'dense')
        assert best_outer == 1
        assert best_inner == max_t

    def test_n_tune_steps_forwarded_to_tune_trial(self, monkeypatch):
        _patch_build_solver(monkeypatch)
        monkeypatch.setattr(_m, 'warmup', lambda q: 0.001)
        observed = []
        def mock_tune_trial(q, n_steps=_N_TUNE_STEPS):
            observed.append(n_steps)
            return 0.05
        monkeypatch.setattr(_m, '_tune_trial', mock_tune_trial)

        tune_threads('Eigen', 'Hilbert', 'dense', n_tune_steps=37)
        assert all(s == 37 for s in observed), (
            f'Expected n_tune_steps=37 for all calls; got {observed}'
        )

    def test_verbose_does_not_crash(self, monkeypatch, capsys):
        self._mock_env(monkeypatch)
        tune_threads('Eigen', 'Hilbert', 'dense', verbose=True)

    def test_verbose_prints_per_pair_output(self, monkeypatch, capsys):
        self._mock_env(monkeypatch)
        tune_threads('Eigen', 'Hilbert', 'dense', verbose=True)
        out = capsys.readouterr().out
        assert 'tune' in out, 'verbose=True must print per-pair tuning output'

    def test_skips_none_qme(self, monkeypatch):
        """build_solver returning None must be silently skipped."""
        def mock_build(engine, space, fmt, solver='lsrk4', unrolling=True,
                       n_outer_threads=1, n_inner_threads=1):
            if n_outer_threads > 1:
                return None
            return _fake_qme(n_outer_threads, n_inner_threads)
        monkeypatch.setattr(_m, 'build_solver', mock_build)
        monkeypatch.setattr(_m, 'warmup', lambda q: 0.001)
        monkeypatch.setattr(_m, '_tune_trial',
                            lambda q, n_steps=_N_TUNE_STEPS: 0.05)

        best_outer, best_inner, _ = tune_threads('Eigen', 'Hilbert', 'dense')
        assert best_outer == 1  # only valid pair

    def test_n_trials_controls_tune_trial_calls(self, monkeypatch):
        _patch_build_solver(monkeypatch)
        monkeypatch.setattr(_m, 'warmup', lambda q: 0.001)
        call_count = [0]
        def mock_tune_trial(q, n_steps=_N_TUNE_STEPS):
            call_count[0] += 1
            return 0.05
        monkeypatch.setattr(_m, '_tune_trial', mock_tune_trial)

        from pyheom._auto import _thread_pair_candidates
        n_pairs = len(_thread_pair_candidates())

        tune_threads('Eigen', 'Hilbert', 'dense', n_trials=3)
        # Each pair gets n_trials calls
        assert call_count[0] == n_pairs * 3


# ---------------------------------------------------------------------------
# auto_select: structure and option forwarding
# ---------------------------------------------------------------------------

class TestAutoSelectLogic:

    def _mock_env(self, monkeypatch, engines=('Eigen',), n_spaces=3, n_formats=2):
        """Fully mocked environment: no real solver runs."""
        monkeypatch.setattr(_m, 'available_engines', lambda: list(engines))
        monkeypatch.setattr(_m, '_gpu_free_bytes', lambda: None)
        _patch_build_solver(monkeypatch)
        monkeypatch.setattr(_m, 'measure_rss_delta', lambda fn, *a, **kw: 0)
        monkeypatch.setattr(_m, 'warmup', lambda q: 0.001)
        monkeypatch.setattr(_m, 'run_trial', lambda q, **kw: 0.05)
        monkeypatch.setattr(_m, 'tune_threads', lambda *a, **kw: (1, 1, 0.001))

    def test_returns_list(self, monkeypatch):
        self._mock_env(monkeypatch)
        assert isinstance(auto_select(tune=False, verbose=False), list)

    def test_non_empty_when_engine_available(self, monkeypatch):
        self._mock_env(monkeypatch)
        results = auto_select(tune=False, verbose=False)
        assert len(results) > 0

    def test_required_keys_present(self, monkeypatch):
        self._mock_env(monkeypatch)
        required = {'engine', 'space', 'format', 'solver', 'unrolling',
                    'n_outer_threads', 'n_inner_threads', 'elapsed', 'rss_delta_mb'}
        for r in auto_select(tune=False, verbose=False):
            missing = required - r.keys()
            assert not missing, f'Missing keys in result: {missing}'

    def test_first_result_marked_recommended(self, monkeypatch):
        self._mock_env(monkeypatch)
        results = auto_select(tune=False, verbose=False)
        assert results[0].get('recommended') is True

    def test_only_first_result_recommended(self, monkeypatch):
        self._mock_env(monkeypatch)
        results = auto_select(tune=False, verbose=False)
        for r in results[1:]:
            assert 'recommended' not in r

    def test_results_sorted_by_elapsed(self, monkeypatch):
        self._mock_env(monkeypatch)
        counter = [0]
        def mock_run_trial(qme, **kw):
            counter[0] += 1
            return 1.0 / counter[0]  # first call slowest
        monkeypatch.setattr(_m, 'run_trial', mock_run_trial)

        results = auto_select(tune=False, verbose=False)
        times = [r['elapsed'] for r in results]
        assert times == sorted(times)

    def test_tune_false_does_not_call_tune_threads(self, monkeypatch):
        self._mock_env(monkeypatch)
        called = []
        monkeypatch.setattr(_m, 'tune_threads',
                            lambda *a, **kw: called.append(1) or (1, 1, 0.001))
        auto_select(tune=False, verbose=False)
        assert called == []

    def test_tune_true_calls_tune_threads_for_eigen(self, monkeypatch):
        self._mock_env(monkeypatch)
        called = []
        monkeypatch.setattr(_m, 'tune_threads',
                            lambda *a, **kw: called.append(1) or (1, 1, 0.001))
        auto_select(tune=True, verbose=False)
        assert len(called) > 0

    def test_n_tune_steps_forwarded_to_tune_threads(self, monkeypatch):
        self._mock_env(monkeypatch)
        observed = []
        def mock_tune_threads(*a, n_tune_steps=_N_TUNE_STEPS, **kw):
            observed.append(n_tune_steps)
            return (1, 1, 0.001)
        monkeypatch.setattr(_m, 'tune_threads', mock_tune_threads)

        auto_select(tune=True, verbose=False, n_tune_steps=99)
        assert all(s == 99 for s in observed)

    def test_engine_singular_overrides_plural(self, monkeypatch):
        self._mock_env(monkeypatch, engines=('Eigen', 'MKL'))
        results = auto_select(engine='Eigen', engines=['MKL'],
                              tune=False, verbose=False)
        assert all(r['engine'] == 'Eigen' for r in results)

    def test_space_filter(self, monkeypatch):
        self._mock_env(monkeypatch)
        results = auto_select(spaces=['Liouville'], tune=False, verbose=False)
        assert all(r['space'] == 'Liouville' for r in results)

    def test_format_filter(self, monkeypatch):
        self._mock_env(monkeypatch)
        results = auto_select(formats=['dense'], tune=False, verbose=False)
        assert all(r['format'] == 'dense' for r in results)

    def test_elapsed_matches_median_of_trials(self, monkeypatch):
        self._mock_env(monkeypatch)
        trial_times = [0.1, 0.3, 0.2]
        call_count = [0]
        def mock_run_trial(qme, **kw):
            t = trial_times[call_count[0] % len(trial_times)]
            call_count[0] += 1
            return t
        monkeypatch.setattr(_m, 'run_trial', mock_run_trial)

        results = auto_select(tune=False, verbose=False, n_trials=3,
                              spaces=['Liouville'], formats=['dense'])
        # median of [0.1, 0.3, 0.2] = 0.2
        assert abs(results[0]['elapsed'] - float(np.median(trial_times))) < 1e-9

    def test_unrollings_false_filters(self, monkeypatch):
        """Passing unrollings=[False] must produce only unrolling=False results."""
        self._mock_env(monkeypatch)
        results = auto_select(engine='Eigen', unrollings=[False],
                              tune=False, verbose=False)
        assert all(r['unrolling'] is False for r in results)

    def test_rss_delta_mb_nonnegative(self, monkeypatch):
        self._mock_env(monkeypatch)
        results = auto_select(tune=False, verbose=False)
        assert all(r['rss_delta_mb'] >= 0 for r in results)


# ---------------------------------------------------------------------------
# Integration tests: actual HEOMSolver (slow, require pylibheom)
# ---------------------------------------------------------------------------

class TestIntegrationAutoSelect:
    """Run auto_select with the real backend; tune=False to keep it fast."""

    def test_returns_valid_results(self):
        results = auto_select(
            engines=['Eigen'], spaces=['Liouville'], formats=['dense'],
            unrollings=[True], tune=False, n_trials=1, verbose=False,
        )
        assert len(results) > 0
        r = results[0]
        assert r['engine'] == 'Eigen'
        assert r['space'] == 'Liouville'
        assert r['format'] == 'dense'
        assert r['elapsed'] > 0
        assert r['elapsed'] < 60  # must finish in under a minute

    def test_recommended_on_fastest(self):
        results = auto_select(
            engines=['Eigen'], spaces=['Liouville', 'ADO'],
            formats=['dense'], tune=False, n_trials=1, verbose=False,
        )
        assert results[0].get('recommended') is True
        times = [r['elapsed'] for r in results]
        assert times == sorted(times)


class TestIntegrationTuneThreads:
    """tune_threads with real solver: checks correctness of returned config."""

    def test_liouville_dense_returns_valid_config(self):
        best_outer, best_inner, best_t = tune_threads(
            'Eigen', 'Liouville', 'dense', n_trials=1, n_tune_steps=5,
        )
        assert best_outer >= 1
        assert best_inner >= 1
        assert 0 < best_t < 10

    def test_winner_is_from_candidate_set(self):
        from pyheom._auto import _thread_pair_candidates
        candidates = _thread_pair_candidates()
        best_outer, best_inner, _ = tune_threads(
            'Eigen', 'Liouville', 'dense', n_trials=1, n_tune_steps=5,
        )
        assert (best_outer, best_inner) in candidates

    def test_n_tune_steps_respected(self):
        """Verify that a very short n_tune_steps run still finishes quickly."""
        import time
        start = time.perf_counter()
        tune_threads('Eigen', 'Liouville', 'dense', n_trials=1, n_tune_steps=2)
        elapsed = time.perf_counter() - start
        # With only 2 steps, even with many pairs it should finish in a few seconds
        assert elapsed < 60

    def test_verbose_produces_output(self, capsys):
        tune_threads('Eigen', 'Liouville', 'dense',
                     n_trials=1, n_tune_steps=2, verbose=True)
        out = capsys.readouterr().out
        assert 'tune' in out
