#  -*- mode:python -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

import enum
from sys import stderr, exit

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

units = {'energy':unit.dimensionless,
         'time':  unit.dimensionless}

def calc_unit():
    if (units['energy'] == unit.dimensionless or units['time'] == unit.dimensionless):
        if (units['energy'] == unit.dimensionless and units['time'] == unit.dimensionless):
            result = 1.0
        else:
            print('[Error] Unit mismatch error: Both unit_energy and unit_time should be dimensionless.', file=stderr)
            exit(1)
    else:
        result = (UNIT_ENERGY_VALUE__J[units['energy']]
                *UNIT_TIME_VALUE__S[units['time']]
                /hbar__J_s)
    return result
