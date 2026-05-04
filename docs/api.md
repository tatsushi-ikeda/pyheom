# API Reference

## Solvers

### `HEOMSolver`

```python
HEOMSolver(H, noises, *, space='hilbert', format='dense', engine='eigen',
            liouville_order='C', solver='lsrk4', unrolling=True, n_tiers,
            n_inner_threads=<auto>, n_outer_threads=1,
            units=None, device=None)
```

Hierarchical equations of motion (HEOM) solver.

**Parameters**

- `H`: system Hamiltonian, shape `(n, n)`, complex128
- `noises`: list of `BathCorrelation` instances (one per bath coupling)
- `space`: `'hilbert'` or `'liouville'`
- `format`: `'dense'` or `'sparse'`
- `engine`: `'eigen'` (default), `'mkl'`, or `'cuda'`
- `liouville_order`: `'C'` (row-major) or `'F'` (column-major)
- `solver`: time integrator: `'rk4'`, `'lsrk4'` (default), or `'rkdp'`
- `unrolling`: enable compile-time static template for `n_level` (default: `True`);
  effective only for `engine='eigen'` with `n_level` in {2, 3, 4}; ignored by MKL and CUDA
- `n_tiers`: hierarchy truncation depth (required)
- `n_inner_threads`: threads for inner matrix operations
  (`Eigen::setNbThreads` / `mkl_set_num_threads`);
  default: `OMP_NUM_THREADS` env var if set, otherwise `cpu_count()`.
  Controls the primary parallelism for all spaces and engines
  (the only parallelism for ADO and CUDA).
- `n_outer_threads`: threads for the OMP outer loop over hierarchy nodes
  (hilbert and liouville spaces only; ignored by ADO and CUDA); default: 1.
  Set to `OMP_NUM_THREADS` or `cpu_count()` to enable node-level parallelism.
- `units`: dict with `'energy'` and/or `'time'` keys, e.g. `{'energy': unit.wavenumber, 'time': unit.femtosecond}`
- `device`: CUDA device index; only valid when `engine='cuda'`

**Properties**

- `rho`: physical density matrix at current time, shape `(n, n)` (zero-copy view)
- `rho_hierarchy`: full ADO array, shape `(storage_size, n, n)` (zero-copy view)
- `storage_size`: total number of ADOs (hierarchy nodes + 1)

**Methods:** see `QMESolver` below.

---

### `RedfieldSolver`

```python
RedfieldSolver(H, noises, *, space='hilbert', format='dense', engine='eigen',
                liouville_order='C', solver='lsrk4', unrolling=True,
                n_inner_threads=1, n_outer_threads=None,
                units=None, device=None)
```

Redfield master equation solver (Born-Markov approximation, no ADO hierarchy).
Parameters are the same as `HEOMSolver` except `n_tiers` is absent.

---

### `QMESolver` (abstract base)

Both solvers share the following interface.

#### `solve`

```python
result = qme.solve(rho_0, t_list, callback=None, e_ops=None, dt=None)
```

Evolve the system and return a `Result`.

- `rho_0`: initial density matrix, shape `(n, n)` or `(storage_size, n, n)`.
- `t_list`: sequence of output times (user units)
- `callback`: called with the current time at each output step
- `e_ops`: list of operators; `Tr(O @ rho(t))` is recorded at every output time
- `dt`: integration step size (user units; required for fixed-step solvers)

Returns a `Result` object.

#### `init`

```python
integrator = qme.init(rho_0, dt)
```

Return a low-level `Integrator` for step-by-step evolution (see below).

#### `auto` (classmethod)

```python
solver        = HEOMSolver.auto(H, noises, n_tiers=5)
solver, info  = HEOMSolver.auto(H, noises, n_tiers=5, return_info=True)

solver        = RedfieldSolver.auto(H, noises)
```

Automatically select the fastest engine, representation space, matrix format,
and ODE solver for the given system by running short timing trials.
Returns a fully constructed solver instance ready for use.

Valid spaces are restricted to those supported by the calling class:
`HEOMSolver` searches hilbert, liouville, and ado;
`RedfieldSolver` searches hilbert and liouville only.

**Parameters**

- `H`, `noises`: same as the constructor
- `engines`: engines to consider; default: all compiled engines
- `spaces`: spaces to consider; default: all valid spaces for this class
- `formats`: formats to consider; default: `['dense', 'sparse']`
- `solvers`: ODE solvers to consider; default: `['lsrk4', 'rk4', 'rkdp']`
- `dt`: step size for timing trials (default: `2.5e-3`)
- `n_warmup_steps`: steps in each warmup call (default: `5`)
- `n_timing_steps`: steps in each timing call (default: `20`)
- `n_trials`: timing trials per configuration; median is used (default: `3`)
- `tune`: if `True`, sweep `(n_outer_threads, n_inner_threads)` pairs for
  eigen/mkl engines to find the fastest thread configuration (default: `False`)
- `verbose`: print one line per configuration tried (default: `False`)
- `return_info`: if `True`, return `(solver, info_dict)` instead of just the solver
- `**kwargs`: forwarded to the constructor (e.g. `n_tiers` for `HEOMSolver`)

**Returns**

The best solver instance, or `(instance, info)` when `return_info=True`.
`info` is a dict with keys `engine`, `space`, `format`, `solver`,
`n_outer_threads`, `n_inner_threads`, and `elapsed` (median timing trial time in seconds).

**Example**

```python
# Let pyheom choose the fastest configuration for this machine and system
qme = HEOMSolver.auto(H, [corr], n_tiers=5)
result = qme.solve(rho_0, t_list, dt=0.0025)

# Inspect what was chosen
qme, info = HEOMSolver.auto(H, [corr], n_tiers=5, return_info=True, verbose=True)
print(info)  # {'engine': 'mkl', 'space': 'ado', 'format': 'sparse', ...}
```

---

## Return types

### `Result`

Returned by `solve()`.

- `result.times`: ndarray of shape `(n_times,)`, the output times
- `result.expect`: list of complex ndarrays of shape `(n_times,)`,
  one per operator in `e_ops`; empty list when `e_ops` is not given

### `Integrator`

Returned by `init()`.  Shares the internal buffer with the parent solver.

```python
integrator = qme.init(rho_0, dt=0.0025)
integrator.advance_to(5.0)          # evolve to t = 5.0
snapshot = integrator.rho_hierarchy.copy()
integrator.advance_to(10.0)
```

- `integrator.t`: current simulation time (user units)
- `integrator.rho`: physical density matrix (zero-copy view)
- `integrator.rho_hierarchy`: full ADO array (zero-copy view); call `.copy()` before the next `advance_to` to preserve
- `integrator.advance_to(t)`: advance to time *t*; raises `ValueError` if `t < current_t`

---

## Bath correlation

### `noise_decomposition`

```python
corr = noise_decomposition(J, T, type_ltc, *, n_psd=None, type_psd=None, n_msd=None)
corr.V = V   # attach system-bath coupling operator
```

Decompose the bath correlation function of spectral density `J` at temperature `T`
into a sum of exponential modes.

- `J`: `SpectralDensity` instance
- `T`: temperature
- `type_ltc`: low-temperature correction method:
  - `'none'`: spectral density poles only
  - `'msd'`: Matsubara spectral decomposition (`n_msd` terms)
  - `'psd'`: Pade spectral decomposition (`n_psd` terms, `type_psd` variant)
  - `'psd+fsd'`: Pade + Fano spectral decomposition

Returns a `BathCorrelation` instance.

### `BathCorrelation`

Dataclass holding the decomposed bath correlation function.
Returned by `noise_decomposition`; must have `.V` set before passing to a solver.

- `gamma`: decay rates, sparse matrix of shape `(K, K)`
- `sigma`: coupling strengths, ndarray of shape `(K,)`
- `phi_0`: initial phases, ndarray of shape `(K,)`
- `s_mat`: real part matrix, sparse `(K, K)`
- `a_mat`: imaginary part matrix, sparse `(K, K)`
- `s_delta`: Markovian (delta-function) term, scalar; default `0.0`
- `V`: system-bath coupling operator, shape `(n, n)`; must be set by the user

---

## Spectral densities

All spectral density classes inherit from `SpectralDensity` and implement
`spectrum(omega)` and `get_poles()`.

### `Drude`

```python
Drude(eta, gamma_c)
```

Drude (Ohmic with Lorentz cutoff) spectral density:
J(w) = eta*gamma_c^2 * w / (w^2 + gamma_c^2).

### `Brown`

```python
Brown(lambda_0, gamma_c, omega_0)
```

Brownian-oscillator spectral density; overdamped/underdamped/critically damped
depending on gamma_c versus 2*omega_0.

### `OverdampedBrown`

```python
OverdampedBrown(lambda_0, gamma_c, omega_0)
```

Alias for the overdamped Brownian oscillator, parameterized to match the Drude form.

### `BrownDrude`

```python
BrownDrude(lambda_0, zeta, gamma_c, omega_0)
```

Combined Brownian-oscillator + Drude spectral density.

---

## Unit system

```python
from pyheom import unit

unit.wavenumber    # energy in cm^-1
unit.electronvolt  # energy in eV
unit.femtosecond   # time in fs
unit.picosecond    # time in ps
unit.dimensionless # natural units (default)
```

Pass as the `units=` argument to the solver constructor:

```python
qme = HEOMSolver(H, [corr], ..., units={'energy': unit.wavenumber,
                                         'time':   unit.femtosecond})
```

---

## Low-level functions

These are used internally by `noise_decomposition` and are rarely needed directly.

```python
from pyheom.noise_decomposition import calc_noise_time_domain, calc_noise_params
from pyheom.pade_spectral_decomposition import psd
from pyheom.summation_over_poles import calc_a_from_poles, calc_s_from_poles
```

- `calc_noise_time_domain(J, T, type_ltc, **kwargs)`: compute (S, A) matrices from spectral density
- `calc_noise_params(S, A)`: convert (S, A) to `BathCorrelation` fields
- `psd(n, type)`: compute Pade poles and weights for the Bose-Einstein function
- `calc_a_from_poles(gamma_k, a_k, t)`: evaluate imaginary part of bath correlation at times t
- `calc_s_from_poles(gamma_k, s_k, t)`: evaluate real part of bath correlation at times t
