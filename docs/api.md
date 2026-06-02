# API Reference

## Solvers

### `HEOMSolver`

```python
HEOMSolver(H, noises, *, space='Hilbert', format='dense', engine='Eigen',
            liouville_order='C', solver='lsrk4', unrolling=True, truncation_depth,
            n_inner_threads=<auto>, n_outer_threads=1,
            units=None, device=None)
```

Hierarchical equations of motion (HEOM) solver.

**Parameters**

- `H`: system Hamiltonian, shape `(n, n)`, complex128
- `noises`: list of `BathCorrelation` instances (one per bath coupling)
- `space`: `'Hilbert'`, `'Liouville'`, or `'ADO'`
- `format`: `'dense'` or `'sparse'`
- `engine`: `'Eigen'` (default), `'MKL'`, or `'CUDA'`
- `liouville_order`: `'C'` (row-major) or `'F'` (column-major)
- `solver`: time integrator: `'rk4'`, `'lsrk4'` (default), or `'rkdp'`
- `unrolling`: enable compile-time static template for `n_level` (default: `True`);
  effective only for `engine='Eigen'` with `n_level` in {2, 3, 4}; ignored by MKL and CUDA
- `truncation_depth`: hierarchy truncation depth (required)
- `n_inner_threads`: threads for inner matrix operations
  (`Eigen::setNbThreads` / `mkl_set_num_threads`);
  default: `OMP_NUM_THREADS` env var if set, otherwise `cpu_count()`.
  Controls the primary parallelism for all spaces and engines
  (the only parallelism for ADO and CUDA).
- `n_outer_threads`: threads for the OMP outer loop over hierarchy nodes
  (Hilbert and Liouville spaces only; ignored by ADO and CUDA); default: 1.
  Set to `OMP_NUM_THREADS` or `cpu_count()` to enable node-level parallelism.
- `units`: dict with `'energy'` and/or `'time'` keys, e.g. `{'energy': unit.wavenumber, 'time': unit.femtosecond}`
- `device`: CUDA device index; only valid when `engine='CUDA'`

**Properties**

- `rho`: physical density matrix at current time, shape `(n, n)` (zero-copy view)
- `rho_hierarchy`: full ADO array, shape `(storage_size, n, n)` (zero-copy view)
- `storage_size`: total number of ADOs (hierarchy nodes + 1)

**Methods:** see `QMESolver` below.

---

### `RedfieldSolver`

```python
RedfieldSolver(H, noises, *, space='Hilbert', format='dense', engine='Eigen',
                liouville_order='C', solver='lsrk4', unrolling=True,
                n_inner_threads=1, n_outer_threads=None,
                units=None, device=None)
```

Redfield master equation solver (Born-Markov approximation, no ADO hierarchy).
Parameters are the same as `HEOMSolver` except `truncation_depth` is absent.

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
solver        = HEOMSolver.auto(H, noises, truncation_depth=5)
solver, info  = HEOMSolver.auto(H, noises, truncation_depth=5, return_info=True)

solver        = RedfieldSolver.auto(H, noises)
```

Automatically select the fastest engine, representation space, matrix format,
and ODE solver for the given system by running short timing trials.
Returns a fully constructed solver instance ready for use.

Valid spaces are restricted to those supported by the calling class:
`HEOMSolver` searches Hilbert, Liouville, and ADO;
`RedfieldSolver` searches Hilbert and Liouville only.

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
  Eigen/MKL engines to find the fastest thread configuration (default: `False`)
- `verbose`: print one line per configuration tried (default: `False`)
- `return_info`: if `True`, return `(solver, info_dict)` instead of just the solver
- `**kwargs`: forwarded to the constructor (e.g. `truncation_depth` for `HEOMSolver`)

**Returns**

The best solver instance, or `(instance, info)` when `return_info=True`.
`info` is a dict with keys `engine`, `space`, `format`, `solver`,
`n_outer_threads`, `n_inner_threads`, and `elapsed` (median timing trial time in seconds).

**Example**

```python
# Let pyheom choose the fastest configuration for this machine and system
qme = HEOMSolver.auto(H, [corr], truncation_depth=5)
result = qme.solve(rho_0, t_list, dt=0.0025)

# Inspect what was chosen
qme, info = HEOMSolver.auto(H, [corr], truncation_depth=5, return_info=True, verbose=True)
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
  - `'msd'`: Matsubara spectrum decomposition (`n_msd` terms)
  - `'psd'`: Pade spectrum decomposition (`n_psd` terms, `type_psd` variant)
  - `'psd+fsd'`: Pade + Fano spectrum decomposition

Returns a `BathCorrelation` instance.

Pade spectrum decomposition:
- J. Hu, M. Luo, F. Jiang, R.-X. Xu, and Y. Yan, *J. Chem. Phys.* **134**, 244106 (2011), https://doi.org/10.1063/1.3602466.

Fano spectrum decomposition:
- L. Cui, H.-D. Zhang, X. Zheng, R.-X. Xu, and Y. Yan, *J. Chem. Phys.* **151**, 024110 (2019), https://doi.org/10.1063/1.5096945;
- H.-D. Zhang, L. Cui, H. Gong, R.-X. Xu, X. Zheng, and Y. Yan, *J. Chem. Phys.* **152**, 064107 (2020), https://doi.org/10.1063/1.5136093.

### `BathCorrelation`

Dataclass holding the decomposed bath correlation function.
Can be returned by `noise_decomposition` or constructed directly (see below).
Must have `.V` set before passing to a solver.

- `gamma`: decay-rate matrix, sparse `(K, K)`; diagonal entries are the pole positions `gamma_k`
- `sigma`: coupling vector, ndarray `(K,)`; use `ones(K)` for independent modes
- `phi_0`: initial-value vector, ndarray `(K,)`; use `ones(K)` for independent modes
- `s_mat`: symmetric correlation coefficient matrix, sparse `(K, K)`
- `a_mat`: antisymmetric correlation coefficient matrix, sparse `(K, K)`
- `s_delta`: Markovian (delta-function) term, scalar; default `0.0`
- `V`: system-bath coupling operator, shape `(n, n)`; must be set by the user

See T. Ikeda and G. D. Scholes, *J. Chem. Phys.* **152**, 204101 (2020), https://doi.org/10.1063/5.0007327 (arXiv: https://arxiv.org/abs/2003.06134).

**Direct construction**

Use this path when the decomposition is already known.  The general formula is:

```
C(t) = phi_0^T * expm(-gamma * t) * (s_mat - i*a_mat) * sigma  +  s_delta * delta(t)
```

For K independent exponential modes (gamma and coefficient matrices diagonal,
phi_0 = sigma = ones(K)), this reduces to:

```
C(t) = sum_k (s_k - i*a_k) * exp(-gamma_k * t)  +  s_delta * delta(t)
```

```python
import numpy as np
import scipy.sparse
from pyheom import BathCorrelation

K       = 2
gamma_k = np.array([1.0+0j, 0.5+1j])  # complex pole positions allowed
s_k     = np.array([0.5,    0.3   ])   # Re(C) coefficients
a_k     = np.array([0.2,    0.1   ])   # Im(C) coefficients

corr = BathCorrelation(
    gamma   = scipy.sparse.diags(gamma_k).tolil(),
    sigma   = np.ones(K, dtype=complex),
    phi_0   = np.ones(K, dtype=complex),
    s_mat   = scipy.sparse.diags(s_k.astype(complex)).tolil(),
    a_mat   = scipy.sparse.diags(a_k.astype(complex)).tolil(),
    s_delta = 0.0,
    V       = V,
)
```

For degenerate (higher-order) poles, any valid coupled decomposition of `gamma`,
`s_mat`, and `a_mat` works.  `noise_decomposition` computes one automatically
from the spectral density.

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

### Custom spectral density

Subclass `SpectralDensity` and implement `spectrum` and `get_poles`:

```python
from pyheom import SpectralDensity, noise_decomposition

class MySpectral(SpectralDensity):
    def __init__(self, lam, gamma_c):
        self.lam     = lam
        self.gamma_c = gamma_c
        self.poles   = self.get_poles()   # must be set in __init__

    def spectrum(self, omega):
        return 2*self.lam*self.gamma_c*omega / (omega**2 + self.gamma_c**2)

    def get_poles(self):
        # Each entry [a, b, m, n] encodes  b * omega^(2n+1) / (a^2 + omega^2)^m.
        # First-order Lorentzian (m=1, n=0): b * omega / (a^2 + omega^2).
        return [[self.gamma_c, 2*self.lam*self.gamma_c, 1, 0]]

corr = noise_decomposition(MySpectral(0.1, 0.5), T=1.0, type_ltc='none')
corr.V = V
```

**Pole format** for `get_poles()`: a list of `[a, b, m, n]` entries where each entry
contributes `b * omega^(2n+1) / (a^2 + omega^2)^m` to `J(omega)`.
Most spectral densities decompose into first-order Lorentzians (`m=1, n=0`).
Complex `a` is allowed for underdamped oscillators.

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
from pyheom.noise_decomposition import calc_bath_corr_poles
from pyheom.pade_decomposition import psd
from pyheom.summation_over_poles import calc_a_from_poles, calc_s_from_poles
```

- `calc_bath_corr_poles(J, T, type_ltc, **kwargs)`: compute (S, A) pole dicts from spectral density
- `psd(n, type)`: compute Pade poles and weights for the Bose-Einstein function
- `calc_a_from_poles(poles)`: compute antisymmetric bath correlation poles A(t)
- `calc_s_from_poles(sd_poles, be_poles)`: compute symmetric bath correlation poles S(t)
