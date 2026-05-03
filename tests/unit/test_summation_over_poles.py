"""Tests for pyheom.summation_over_poles: calc_a_from_poles and calc_s_from_poles.

Each test case is chosen so the result can be verified analytically:
  A(t) = -(1/pi) int_0^inf f(w) sin(wt) dw
  S(t) = (2/pi) int_0^inf f(w) g(w) cos(wt) dw
where f and g are pole-sum representations.
"""

import numpy as np
import pytest

from pyheom.summation_over_poles import calc_a_from_poles, calc_s_from_poles

pytestmark = pytest.mark.unit


def eval_poles(result, t):
    """Evaluate sum c * t^l * exp(-a*t), skipping delta terms (a=inf)."""
    return sum(
        c * t**l * np.exp(-a * t)
        for (a, l), c in result.items()
        if a != np.inf
    )


# ---------------------------------------------------------------------------
# calc_a_from_poles
# ---------------------------------------------------------------------------

class TestCalcAFromPoles:
    """
    For a single first-order pole [a, b, 1, 0]:
        A(t) = -(b/2) * exp(-a*t)     [standard residue integral]

    For a second-order pole [a, b, 2, 0]:
        A(t) = -(b/(4a)) * t * exp(-a*t)

    For [a, b, 2, 1]:
        A(t) = -(b/2)*exp(-at) + (b*a^2/4)*t*exp(-at)
    """

    def test_single_first_order_pole_coefficients(self):
        # [[1.0, 2.0, 1, 0]] -> {(1.0, 0): -1.0}
        result = calc_a_from_poles([[1.0, 2.0, 1, 0]])
        assert result[(1.0, 0)] == pytest.approx(-1.0, rel=1e-12)

    @pytest.mark.parametrize("t,expected", [
        (0.5, -np.exp(-0.5)),
        (1.0, -np.exp(-1.0)),
        (2.0, -np.exp(-2.0)),
    ])
    def test_single_first_order_pole_time_series(self, t, expected):
        # A(t) = -exp(-t): -(2/2)*exp(-1*t)
        result = calc_a_from_poles([[1.0, 2.0, 1, 0]])
        assert eval_poles(result, t) == pytest.approx(expected, rel=1e-12)

    def test_second_order_pole_coefficients(self):
        # [[2.0, 8.0, 2, 0]] -> l=1 term only: -(8/(4*2))*t*exp(-2t) = -t*exp(-2t)
        result = calc_a_from_poles([[2.0, 8.0, 2, 0]])
        assert result[(2.0, 1)] == pytest.approx(-1.0, rel=1e-12)

    @pytest.mark.parametrize("t,expected", [
        (0.5, -0.5 * np.exp(-1.0)),
        (1.0, -np.exp(-2.0)),
        (2.0, -2.0 * np.exp(-4.0)),
    ])
    def test_second_order_pole_time_series(self, t, expected):
        # A(t) = -t*exp(-2t): -(8/(4*2))*t*exp(-2t)
        result = calc_a_from_poles([[2.0, 8.0, 2, 0]])
        assert eval_poles(result, t) == pytest.approx(expected, rel=1e-12)

    def test_two_first_order_poles_coefficients(self):
        # [[1,2,1,0],[3,6,1,0]] -> -(b/2)*exp(-at) for each
        result = calc_a_from_poles([[1.0, 2.0, 1, 0], [3.0, 6.0, 1, 0]])
        assert result[(1.0, 0)] == pytest.approx(-1.0, rel=1e-12)
        assert result[(3.0, 0)] == pytest.approx(-3.0, rel=1e-12)

    @pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
    def test_two_first_order_poles_time_series(self, t):
        # A(t) = -exp(-t) - 3*exp(-3t)
        result = calc_a_from_poles([[1.0, 2.0, 1, 0], [3.0, 6.0, 1, 0]])
        expected = -np.exp(-t) - 3 * np.exp(-3 * t)
        assert eval_poles(result, t) == pytest.approx(expected, rel=1e-12)

    def test_second_order_pole_n1_coefficients(self):
        # [[1.0, 1.0, 2, 1]] -> {(1,0): -0.5, (1,1): 0.25}
        # A(t) = -(b/2)*exp(-at) + (b*a^2/4)*t*exp(-at) for a=1,b=1
        result = calc_a_from_poles([[1.0, 1.0, 2, 1]])
        assert result[(1.0, 0)] == pytest.approx(-0.5, rel=1e-12)
        assert result[(1.0, 1)] == pytest.approx(0.25, rel=1e-12)

    @pytest.mark.parametrize("t,expected", [
        (0.5, -0.375 * np.exp(-0.5)),    # (-0.5 + 0.25*0.5)*exp(-0.5)
        (1.0, -0.25 * np.exp(-1.0)),     # (-0.5 + 0.25)*exp(-1)
        (3.0, (-0.5 + 0.75) * np.exp(-3.0)),
    ])
    def test_second_order_pole_n1_time_series(self, t, expected):
        result = calc_a_from_poles([[1.0, 1.0, 2, 1]])
        assert eval_poles(result, t) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# calc_s_from_poles
# ---------------------------------------------------------------------------

class TestCalcSFromPoles:
    """
    Analytic results for S(t) = (2/pi) int_0^inf f(w) g(w) cos(wt) dw:

    Same-pole case  [[a,b,1,0]] x [[a,b',1,0]] -> S(t) = (b*b')/(2a) * (1-at) * exp(-at)
    Cross-pole case [[a,b,1,0]] x [[a',b',1,0]] (a!=a') ->
        S(t) = 2*b*b'/(a'^2-a^2) * [a'*exp(-a't) - a*exp(-at)] / 2   (partial fractions)
    Classical limit [[a,b,1,0]] x [[0,T,1,0]] -> S(t) = (b*T/a) * exp(-at)
    """

    def test_same_pole_coefficients(self):
        # [[1,1,1,0]] x [[1,1,1,0]]: S(t) = 0.5*(1-t)*exp(-t)
        result = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[1.0, 1.0, 1, 0]])
        assert result[(1.0, 0)] == pytest.approx(0.5, rel=1e-12)
        assert result[(1.0, 1)] == pytest.approx(-0.5, rel=1e-12)

    @pytest.mark.parametrize("t,expected", [
        (0.5, 0.5 * (1 - 0.5) * np.exp(-0.5)),   # 0.25*exp(-0.5)
        (1.0, 0.0),                                 # (1-1) = 0
        (2.0, 0.5 * (1 - 2) * np.exp(-2.0)),      # -0.5*exp(-2)
    ])
    def test_same_pole_time_series(self, t, expected):
        result = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[1.0, 1.0, 1, 0]])
        assert eval_poles(result, t) == pytest.approx(expected, abs=1e-12)

    def test_cross_pole_coefficients(self):
        # [[1,1,1,0]] x [[2,1,1,0]]
        # S(t) = -exp(-t)/3 + 2*exp(-2t)/3
        result = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[2.0, 1.0, 1, 0]])
        assert result[(1.0, 0)] == pytest.approx(-1 / 3, rel=1e-12)
        assert result[(2.0, 0)] == pytest.approx(2 / 3, rel=1e-12)

    @pytest.mark.parametrize("t,expected", [
        (0.5, -np.exp(-0.5)/3 + 2*np.exp(-1.0)/3),
        (1.0, -np.exp(-1.0)/3 + 2*np.exp(-2.0)/3),
        (2.0, -np.exp(-2.0)/3 + 2*np.exp(-4.0)/3),
    ])
    def test_cross_pole_time_series(self, t, expected):
        result = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[2.0, 1.0, 1, 0]])
        assert eval_poles(result, t) == pytest.approx(expected, rel=1e-12)

    def test_classical_bose_einstein_coefficients(self):
        # [[1,1,1,0]] x [[0,1,1,0]]: classical limit T=1
        # S(t) = (b*T/a)*exp(-at) = exp(-t)
        result = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[0, 1.0, 1, 0]])
        assert result[(1.0, 0)] == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
    def test_classical_bose_einstein_time_series(self, t):
        result = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[0, 1.0, 1, 0]])
        assert eval_poles(result, t) == pytest.approx(np.exp(-t), rel=1e-12)

    def test_symmetry_cross_poles(self):
        # calc_s_from_poles is symmetric: swapping poles_1 and poles_2 gives same S(t)
        r1 = calc_s_from_poles([[1.0, 1.0, 1, 0]], [[2.0, 1.0, 1, 0]])
        r2 = calc_s_from_poles([[2.0, 1.0, 1, 0]], [[1.0, 1.0, 1, 0]])
        for t in [0.5, 1.0, 2.0]:
            assert eval_poles(r1, t) == pytest.approx(eval_poles(r2, t), rel=1e-12)
