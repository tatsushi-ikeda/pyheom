<p align="center">
    <img src="https://raw.githubusercontent.com/tatsushi-ikeda/libheom/master/etc/libheom_logo_simple.svg" alt="LibHEOM" height=96>
</p>

# PyHEOM: Python Library for Open Quantum Dynamics based on HEOM Theory

`pyheom` is an open-source Python library for open quantum dynamics simulations.
It provides HEOM (Hierarchical Equations of Motion) and Redfield master equation solvers
backed by a C++ core (`libheom`) with CPU (Eigen/MKL) and GPU (CUDA) backends.

The current release is v1.0.0a4.

## Documentation

Full documentation (installation, quick start, API reference) is available in [`docs/`](docs/index.md).

## Requirements

- Python 3.9+
- NumPy, SciPy
- CMake 3.20+, C++17 compiler (Intel icpx 2024+ or GCC 8+)

Optional backends:

- Intel MKL (`mkl` engine)
- CUDA 11.7+ / nvcc (`cuda` engine)

## Installation

```bash
pip install -e .
```

For MKL and CUDA backends see [`docs/installation.md`](docs/installation.md).

This [Google Colaboratory example](https://colab.research.google.com/github/tatsushi-ikeda/pyheom/blob/master/examples/colab/pyheom_example_2level_cpu.ipynb) may be helpful.

<a href="https://colab.research.google.com/github/tatsushi-ikeda/pyheom/blob/master/examples/colab/pyheom_example_2level_cpu.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

## Authors
* **Tatsushi Ikeda** (ikeda.tatsushi.37u@kyoto-u.jp)

## Licence
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

`libheom` and `pyheom` are distributed under the BSD 3-clause License. See the `LICENSE.txt` file for details.
See also [Disclaimer](docs/disclaimer.md) for notes on documentation.

## Citation Information

```Plain Text
@article{ikeda2020jcp,
   author = {Ikeda, Tatsushi and Scholes, Gregory D.},
   title = {Generalization of the hierarchical equations of motion theory for efficient calculations with arbitrary correlation functions},
   journal = {The Journal of Chemical Physics},
   volume = {152},
   number = {20},
   pages = {204101},
   ISSN = {0021-9606},
   DOI = {10.1063/5.0007327},
   url = {https://doi.org/10.1063/5.0007327},
   year = {2020},
   type = {Journal Article}
}
```

## Acknowledgments

<p align="center">
    <a href="https://www.jsps.go.jp/"><img src="https://www.jsps.go.jp/j-grantsinaid/06_jsps_info/g_120612/data/whiteKAKENHIlogoM_jp.jpg" alt="KAKENHI" height=48 hspace=8></a>
    <a href="https://www.moore.org/"><img src="https://www.moore.org/docs/default-source/Grantee-Resources/foundation-logos/moore-logo-color.jpg?sfvrsn=2" alt="MOORE" height=48 hspace=8></a>
</p>

-   A prototype of this library was developed for projects supported by [Japan Society for the Promotion of Science](https://www.jsps.go.jp/).
    The current version is being developed for projects funded by JSPS again.
-   The version for the above research paper (v0.5) was developed in [the Scholes group](http://chemlabs.princeton.edu/scholes/) for projects supported by [the Gordon and Betty Moore Foundation](https://www.moore.org/).
