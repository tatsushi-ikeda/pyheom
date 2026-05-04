"""Unit tests for pyheom._auto: thread candidate generation.

These tests require no compiled backend and cover the pure-Python logic that
selects which (n_outer, n_inner) pairs to probe during thread tuning.

Key regression: _thread_pair_candidates must use cpu_count() as the upper
bound, not OMP_NUM_THREADS.  Intel Python sets OMP_NUM_THREADS=1 by default,
which previously caused auto-tune to test only (1,1) and always report the
single-thread result as optimal.
"""

from multiprocessing import cpu_count

import pytest

from pyheom._auto import _thread_candidates, _thread_pair_candidates, _rss_bytes

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _thread_candidates
# ---------------------------------------------------------------------------

class TestThreadCandidates:

    def test_always_includes_1(self):
        assert 1 in _thread_candidates()

    def test_always_includes_cpu_count(self):
        assert cpu_count() in _thread_candidates()

    def test_no_duplicates(self):
        cands = _thread_candidates()
        assert len(cands) == len(set(cands))

    def test_all_positive(self):
        assert all(n >= 1 for n in _thread_candidates())

    def test_sorted_ascending(self):
        cands = _thread_candidates()
        assert cands == sorted(cands)

    def test_at_least_one_element(self):
        assert len(_thread_candidates()) >= 1

    def test_independent_of_omp_num_threads_eq_1(self, monkeypatch):
        """cpu_count() must appear even when OMP_NUM_THREADS=1."""
        monkeypatch.setenv('OMP_NUM_THREADS', '1')
        import pyheom._auto as m
        cands = m._thread_candidates()
        assert cpu_count() in cands

    def test_multi_core_yields_multiple_candidates(self, monkeypatch):
        """On a multi-core machine we must test more than one value."""
        monkeypatch.setenv('OMP_NUM_THREADS', '1')
        import pyheom._auto as m
        if cpu_count() > 1:
            assert len(m._thread_candidates()) >= 2


# ---------------------------------------------------------------------------
# _thread_pair_candidates
# ---------------------------------------------------------------------------

class TestThreadPairCandidates:

    def test_returns_list(self):
        assert isinstance(_thread_pair_candidates(), list)

    def test_each_element_is_2_tuple(self):
        for p in _thread_pair_candidates():
            assert len(p) == 2

    def test_all_positive(self):
        for n_outer, n_inner in _thread_pair_candidates():
            assert n_outer >= 1
            assert n_inner >= 1

    def test_no_zeros(self):
        assert all(p[0] > 0 and p[1] > 0 for p in _thread_pair_candidates())

    def test_sorted(self):
        pairs = _thread_pair_candidates()
        assert pairs == sorted(pairs)

    def test_no_duplicates(self):
        pairs = _thread_pair_candidates()
        assert len(pairs) == len(set(pairs))

    def test_always_includes_single_thread(self):
        assert (1, 1) in _thread_pair_candidates()

    def test_includes_omp_dominant_pair(self):
        """(cpu_count(), 1) must be present on multi-core machines."""
        if cpu_count() > 1:
            assert (cpu_count(), 1) in _thread_pair_candidates()

    def test_includes_inner_dominant_pair(self):
        """(1, cpu_count()) must be present on multi-core machines."""
        if cpu_count() > 1:
            assert (1, cpu_count()) in _thread_pair_candidates()

    def test_includes_ceiling_pair(self):
        """(cpu_count(), cpu_count()) always included as oversubscription check."""
        assert (cpu_count(), cpu_count()) in _thread_pair_candidates()

    def test_independent_of_omp_num_threads(self, monkeypatch):
        """THE REGRESSION: previously OMP_NUM_THREADS=1 restricted candidates to (1,1).

        After the fix, _thread_pair_candidates uses cpu_count() so multi-core
        pairs are always included regardless of OMP_NUM_THREADS.
        """
        monkeypatch.setenv('OMP_NUM_THREADS', '1')
        import pyheom._auto as m
        pairs = m._thread_pair_candidates()
        if cpu_count() > 1:
            assert any(n > 1 for p in pairs for n in p), (
                'OMP_NUM_THREADS=1 must not restrict thread pair candidates to (1,1); '
                'cpu_count()-based pairs must still be present.'
            )
            assert (cpu_count(), 1) in pairs
            assert (1, cpu_count()) in pairs

    def test_multi_core_has_many_candidates(self, monkeypatch):
        """On a multi-core machine there should be many distinct pairs to try."""
        monkeypatch.setenv('OMP_NUM_THREADS', '1')
        import pyheom._auto as m
        pairs = m._thread_pair_candidates()
        if cpu_count() > 1:
            assert len(pairs) > 1


# ---------------------------------------------------------------------------
# _rss_bytes
# ---------------------------------------------------------------------------

class TestRssBytes:

    def test_returns_nonnegative(self):
        assert _rss_bytes() >= 0

    def test_returns_int(self):
        assert isinstance(_rss_bytes(), int)

    def test_two_calls_are_consistent(self):
        a = _rss_bytes()
        b = _rss_bytes()
        assert a >= 0 and b >= 0
