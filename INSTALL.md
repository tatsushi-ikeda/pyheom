# Installation

## Preparation

First, you need to install the compulsory and optional dependent libraries.
You can use the following command to fetch `libheom`, `Eigen3`, and `pybind11`.

```bash
git submodule update --init --recursive
```

If you want to enable `Intel MKL` and `CUDA` modules in `libheom`, you need to install and activate them before install `pyheom` (See `INSTALL.md` in `libheom`).
Note that when you enable these modules, you need to make sure that `Intel MKL` and/or `CUDA` are activated at runtime. 
Otherwise, you will face import errors such as the following:

```text
ImportError: libmkl_rt.so.1: cannot open shared object file: No such file or directory
```

## Installation

Type the following:

```bash
pip install .
```

or

```bash
pip install git+https://github.com/tatsushi-ikeda/pyheom
```

You can specify arguments for cmake by using the environment variable `CMAKE_ARGS`.
For example,

```bash
CMAKE_ARGS="-DCMAKE_CXX_COMPILER=icc -DCUDA_ARCH_LIST=60 -DCMAKE_VERBOSE_MAKEFILE=ON" pip install -e . -v
```

Verbose modes (`-v` for `pip` and `-DCMAKE_VERBOSE_MAKEFILE=ON` for `cmake`) will help you to specify details of installation.

Regarding possible options in `CMAKE_ARGS`, see `INSTALL.md` in `libheom`

