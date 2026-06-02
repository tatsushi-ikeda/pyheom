# Quick Start

This walkthrough covers a 2-level Brownian-oscillator HEOM simulation.
All quantities are dimensionless.

## System setup

```python
import numpy as np
from pyheom import HEOMSolver, noise_decomposition, Brown

# Physical parameters
lambda_0 = 0.01   # coupling constant
omega_0  = 1.0    # vibrational frequency
zeta     = 0.5    # damping constant
T        = 1.0    # temperature

# Renormalized frequency (shifts H due to the bath)
omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)

H = np.array([[omega_1, 0.0], [0.0, 0.0]], dtype=np.complex128)
V = np.array([[0.0,    1.0], [1.0, 0.0]], dtype=np.complex128)
```

## Bath correlation decomposition

```python
J    = Brown(lambda_0, zeta, omega_0)
corr = noise_decomposition(J, T=T, type_ltc='psd', n_psd=1, type_psd='n-1/n')
corr.V = V   # attach the system-bath coupling operator
```

`type_ltc='psd'` uses the Pade spectrum decomposition for the low-temperature
correction.

## Creating the solver

```python
qme = HEOMSolver(
    H, [corr],
    space='Liouville', format='dense', engine='Eigen',
    liouville_order='C', solver='lsrk4',
    truncation_depth=5, n_inner_threads=1, n_outer_threads=1,
)
```

Use `engine='CUDA'` to run on GPU.

Alternatively, let pyheom benchmark all available engines and configurations
and select the fastest one automatically:

```python
qme = HEOMSolver.auto(H, [corr], truncation_depth=5)
```

See `HEOMSolver.auto()` in the [API reference](api.md) for tuning options.

## Time evolution

```python
rho_0 = np.zeros((2, 2), dtype=np.complex128)
rho_0[0, 0] = 1.0

t_list = np.linspace(0.0, 25.0, 1001)
result = qme.solve(rho_0, t_list, dt=0.0025)

# qme.rho holds the density matrix at the final time
print('rho_00(t=25):', qme.rho[0, 0].real)   # expected ~0.278
```

## Collecting expectation values

Use `e_ops` to record `Tr(O rho(t))` at every output time without a Python callback:

```python
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

result = qme.solve(rho_0, t_list, e_ops=[sigma_z], dt=0.0025)
# result.expect[0] is a complex array of shape (1001,)
```

## Plotting

```python
import matplotlib.pyplot as plt

times = np.linspace(0.0, 25.0, 1001)
rho_0 = np.zeros((2, 2), dtype=np.complex128); rho_0[0, 0] = 1.0
proj0 = np.diag([1.0, 0.0]).astype(np.complex128)
proj1 = np.diag([0.0, 1.0]).astype(np.complex128)

result = qme.solve(rho_0, times, e_ops=[proj0, proj1], dt=0.0025)

plt.plot(times, result.expect[0].real, label=r'$\rho_{00}$')
plt.plot(times, result.expect[1].real, label=r'$\rho_{11}$')
plt.xlabel('time'); plt.ylabel('population')
plt.legend(); plt.tight_layout(); plt.show()
```
