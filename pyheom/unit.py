#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------

import enum

unit = enum.Enum('unit',
 '''dimensionless
    femtosecond
    picosecond
    wavenumber
    electronvolt''')

hbar__J_s = 1.05457180013e-34

UNIT_ENERGY_VALUE__J = {
    unit.wavenumber:   1.98644582441459e-23, # (299792458*100*6.62607004081e-34)
    unit.electronvolt: 1.602176620898e-19,
};
UNIT_TIME_VALUE__S = {
    unit.femtosecond: 1.0e-15,
    unit.picosecond:  1.0e-12,
}


def calc_unit_from_dict(d):
    """Compute the unit conversion factor from a `{'energy': ..., 'time': ...}` dict."""
    energy = d['energy']
    time   = d['time']
    if (energy == unit.dimensionless) != (time == unit.dimensionless):
        raise ValueError(
            "Unit mismatch: 'energy' and 'time' must both be dimensionless or both have units "
            f"(got energy={energy}, time={time})"
        )
    if energy == unit.dimensionless:
        return 1.0
    return (UNIT_ENERGY_VALUE__J[energy]
            * UNIT_TIME_VALUE__S[time]
            / hbar__J_s)
