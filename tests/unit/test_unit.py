"""Tests for pyheom.unit: calc_unit_from_dict() values and ValueError on mismatch.

Numerical-contract style: physical conversion factors are pinned, independent
of any future API renames.
"""

import math

import pytest

from pyheom.unit import (
    unit, calc_unit_from_dict, hbar__J_s, UNIT_ENERGY_VALUE__J, UNIT_TIME_VALUE__S,
)


pytestmark = pytest.mark.unit


class TestCalcUnitFromDict:
    def test_both_dimensionless_returns_one(self):
        assert calc_unit_from_dict(
            {'energy': unit.dimensionless, 'time': unit.dimensionless}
        ) == 1.0

    def test_wavenumber_femtosecond(self):
        expected = (UNIT_ENERGY_VALUE__J[unit.wavenumber]
                    * UNIT_TIME_VALUE__S[unit.femtosecond]
                    / hbar__J_s)
        assert calc_unit_from_dict(
            {'energy': unit.wavenumber, 'time': unit.femtosecond}
        ) == pytest.approx(expected, rel=1e-15)

    def test_electronvolt_picosecond(self):
        expected = (UNIT_ENERGY_VALUE__J[unit.electronvolt]
                    * UNIT_TIME_VALUE__S[unit.picosecond]
                    / hbar__J_s)
        assert calc_unit_from_dict(
            {'energy': unit.electronvolt, 'time': unit.picosecond}
        ) == pytest.approx(expected, rel=1e-15)

    def test_value_is_finite_and_positive(self):
        v = calc_unit_from_dict({'energy': unit.wavenumber, 'time': unit.picosecond})
        assert math.isfinite(v) and v > 0

    @pytest.mark.parametrize("energy,time", [
        (unit.wavenumber,    unit.dimensionless),
        (unit.electronvolt,  unit.dimensionless),
        (unit.dimensionless, unit.femtosecond),
        (unit.dimensionless, unit.picosecond),
    ])
    def test_mismatch_raises_value_error(self, energy, time):
        with pytest.raises(ValueError, match="Unit mismatch"):
            calc_unit_from_dict({'energy': energy, 'time': time})
