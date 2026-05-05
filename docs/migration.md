# Migration Guide

## v0.5 (SI) -> v1.0

This section covers breaking changes from the v0.5 supplementary information code to v1.0.

### Solver class names

```python
# v0.5
import pyheom
solver = pyheom.HEOM(H, noises, max_tier=5, ...)
solver = pyheom.Redfield(H, noises, ...)

# v1.0
from pyheom import HEOMSolver, RedfieldSolver
solver = HEOMSolver(H, noises, n_tiers=5, ...)
solver = RedfieldSolver(H, noises, ...)
```

### Spectral density class names

```python
# v0.5
J = pyheom.Brownian(lambda_0, zeta, omega_0)
J = pyheom.BrownianDrivenByDrudian(...)
J = pyheom.Drudian(...)

# v1.0
from pyheom import Brown, BrownDrude, Drude
J = Brown(lambda_0, zeta, omega_0)
J = BrownDrude(...)
J = Drude(...)
```

### Noise format

```python
# v0.5  (nested dict; noise_decomposition args use uppercase)
corr = pyheom.noise_decomposition(J, T=1, type_LTC='PSD', n_PSD=1, type_PSD='N-1/N')
noise = {"V": V, "C": corr}   # C contains: gamma, sigma, phi_0, s, a, S_delta

# v1.0  (BathCorrelation dataclass; args use lowercase)
from pyheom import noise_decomposition, Brown
corr = noise_decomposition(Brown(...), T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
corr.V = V
solver = HEOMSolver(H, [corr], ...)
```

Key renames from v0.5 `C` dict to `BathCorrelation` attributes:

- `s` -> `s_mat`
- `a` -> `a_mat`
- `S_delta` -> `s_delta`

### Time evolution interface

```python
# v0.5
solver.set_rho(rho_0)
solver.time_evolution(dt, count, callback, callback_interval)

# v1.0
result = solver.solve(rho_0, t_list, callback=callback, dt=dt)
```

`solve()` now returns a `Result` object; `solver.rho` is a zero-copy view.

### Density matrix access

```python
# v0.5
rho   = solver.get_rho()           # copy
rho_h = np.zeros(...)              # caller allocates hierarchy buffer

# v1.0
rho   = solver.rho                 # zero-copy view
rho_h = solver.rho_hierarchy       # zero-copy view of full ADO array
```

### Liouville space ordering

```python
# v0.5: not exposed
# intermediate master (deprecated)
solver = HEOMSolver(..., order_liouville='row_major')

# v1.0
solver = HEOMSolver(..., liouville_order='C')   # or 'F'
```

### Backend and format selection

```python
# v0.5
solver = pyheom.HEOM(..., gpu_device=0, matrix_type='dense',
                     hierarchy_connection='loop')

# v1.0
solver = HEOMSolver(..., engine='CUDA', device=0, format='dense', space='Hilbert')
```

### Known bug in v0.5

`pyheom.py::get_coo_matrix()` has a typo `ipml_class_name += "_c"` (should be
`impl_class_name`), silently breaking the `complex64` dtype path. Fixed in v1.0.

---

## v1.0.0a2 -> v1.0.0b1

These changes apply to users of the intermediate master-branch releases.

### Unit system

The module-level `pyheom.units` global and the no-argument `pyheom.unit.calc_unit()`
have been removed.  Pass `units=` to the solver constructor instead; omit it to
use dimensionless (natural) units.  `pyheom.unit.calc_unit_from_dict` remains
available for computing a conversion factor directly.

```python
# removed (no longer available)
import pyheom
pyheom.units['energy'] = pyheom.unit.wavenumber
pyheom.units['time']   = pyheom.unit.femtosecond

# v1.0.0b1
solver = HEOMSolver(H, [corr], ..., units={'energy': unit.wavenumber,
                                           'time':   unit.femtosecond})
```

### noise_decomposition return type

```python
# v1.0.0a2  (returned a plain dict)
corr = noise_decomposition(J, T=T, type_ltc='none')
solver = HEOMSolver(H, [dict(V=V, **corr)], ...)

# v1.0.0b1  (returns BathCorrelation dataclass)
corr   = noise_decomposition(J, T=T, type_ltc='none')
corr.V = V
solver = HEOMSolver(H, [corr], ...)
```

Dict-style access (`corr['gamma']`) is no longer supported; use `corr.gamma`.

### Spectral density class names

```python
# v1.0.0a2
from pyheom import drude, brown, overdamped_brown, brown_drude

# v1.0.0b1
from pyheom import Drude, Brown, OverdampedBrown, BrownDrude
```

### Solver class names

```python
# v1.0.0a2
from pyheom import heom_solver, redfield_solver
solver = heom_solver(H, [corr], ...)
solver = redfield_solver(H, [corr], ...)

# v1.0.0b1
from pyheom import HEOMSolver, RedfieldSolver
solver = HEOMSolver(H, [corr], ...)
solver = RedfieldSolver(H, [corr], ...)
```

### New in v1.0.0b1

- `solve()` returns `Result(times, expect)`.
- `e_ops=[O_1, ...]` collects expectation values at each output time.
- `solve(rho_0)` accepts `rho_0` of shape `(storage_size, n_level, n_level)` for hierarchy restart.
- `qme.init(rho_0, dt)` returns an `Integrator` for step-by-step evolution.
- `device=` promoted to a top-level solver keyword (was `engine_args={'device': 0}`).
