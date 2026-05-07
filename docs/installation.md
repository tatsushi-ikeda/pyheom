# Installation

## Requirements

- Python 3.9+
- NumPy, SciPy
- CMake 3.20+, C++17 compiler (Intel icpx 2024+ or GCC 8+)

Optional backends:

- Intel MKL (`mkl` engine)
- CUDA 11.7+ / nvcc (`cuda` engine)

## Getting the source

Clone the repository and initialise submodules (libheom, Eigen3, pybind11):

```bash
git clone https://github.com/tatsushi-ikeda/pyheom
cd pyheom
git submodule update --init --recursive
```

## Environment setup (HPC cluster example)

The exact module names depend on your cluster.  The example below is for a
cluster that provides Intel oneAPI and CUDA via the `module` command:

```bash
module load intel-python3/... intel/... cuda/...   # adjust to your HPC environment
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy
```

On systems without a module system, install the Intel oneAPI toolkit and CUDA
toolkit separately and ensure the compilers are on `PATH`.

## Building from source

### Eigen backend (CPU, default)

```bash
pip install -e .
```

### MKL backend

Set `MKLROOT` to the MKL installation directory (loading the Intel oneAPI
module typically exports this automatically), then build:

```bash
CPATH=$MKLROOT/include:$CPATH \
LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH \
CMAKE_ARGS="-DLIBHEOM_USE_MKL=ON \
            -DBLAS_LIBRARIES=$MKLROOT/lib/intel64/libmkl_rt.so" \
pip install -e .
```

The MKL libraries must remain on `LD_LIBRARY_PATH` at runtime; otherwise import
will fail with `ImportError: libmkl_rt.so.1: cannot open shared object file`.

### CUDA backend

```bash
CMAKE_ARGS="-DLIBHEOM_USE_CUDA=ON -DCUDA_ARCH_LIST=70" pip install -e .
```

Replace `70` with the compute capability of your GPU (e.g. `75` for T4, `80` for A100).
The CUDA toolkit must be active at runtime.

## Verifying the build

```python
import pyheom.pylibheom as lb
print('Eigen:', lb.eigen_is_supported())
print('MKL:  ', lb.mkl_is_supported())
print('CUDA: ', lb.cuda_is_supported())
```

## Running the test suite

```bash
python -m pytest tests/
```
