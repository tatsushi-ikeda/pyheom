<p align="center">
    <img src="https://raw.githubusercontent.com/tatsushi-ikeda/libheom/master/etc/libheom_logo_simple.svg" alt="LibHEOM" height=96>
</p>

# PyHEOM: Python 3 Library to Simulate Open Quantum Dynamics based on HEOM Theory

The current stable version is [v0.5](https://github.com/tatsushi-ikeda/pyheom/tree/v0.5). 

Version [1.0 (alpha)](https://github.com/tatsushi-ikeda/pyheom/tree/develop) is under development. 
The Master branch also could be unstable.

## Introduction

`pyheom` is an open-source library that supports open quantum dynamics simulations based on the hierarchical equations of motion (HEOM) theory.
This library provides a python 3 binding of `libheom` (`pylibheom`) and high-level APIs.
All future development will be handled in this repository.

This library is still under development, and some optional functions are not implemented.
There are no guarantees about backward compatibility as of now (Version 0.6).

## TODO

-   Write API documentation
-   Rewrite codes for non-linear spectra calculations which are temporarily removed

## Required Packages

-   Python 3.6 or later
-   libheom and its dependent libraries:
    [https://github.com/tatsushi-ikeda/libheom](https://github.com/tatsushi-ikeda/libheom)
-   pybind11:
    [https://github.com/pybind/pybind11](https://github.com/pybind/pybind11)
-   numpy:
    [https://numpy.org/](https://numpy.org/)
-   scipy:
    [https://www.scipy.org/](https://www.scipy.org/)
-   jinja2:
    [https://jinja.palletsprojects.com](https://jinja.palletsprojects.com/en/3.1.x/)

## Installation

Type the following command from the source tree directory:

```bash
pip install .
```

or

```
pip install git+https://github.com/tatsushi-ikeda/pyheom
```

You can specify arguments for cmake by using the environment variable `CMAKE_ARGS`.
For details, see [INSTALL.md](INSTALL.md).

This [Google Colaboratory example](https://colab.research.google.com/github/tatsushi-ikeda/pyheom/blob/develop/examples/pyheom_example_2level_cpu.ipynb) may be helpful.

<a href="https://colab.research.google.com/github/tatsushi-ikeda/pyheom/blob/develop/examples/pyheom_example_2level_cpu.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

## Authors
* **Tatsushi Ikeda** (ikeda.tatsushi.37u@kyoto-u.jp)

## Licence
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

`libheom` is distributed under the BSD 3-clause License. See the `LICENSE.txt` file for details.

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
