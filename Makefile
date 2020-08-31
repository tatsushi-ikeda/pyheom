SRC     = pyheomcore.cpp \
          hierarchy_structure.cpp \
          Liouvillian.cpp \
          operations.cpp \
          time_evolution.cpp

SRC_GPU = operations_gpu.cu \
	  utilities_gpu.cu

ifeq ($(OS),Windows_NT)
  PYTHON   := python-vs
  EXEEXT   := exe
  OBJEXT   := obj
  LIBEXT   := lib
  SHAREEXT := pyd

  CXX	    := cl
  CXXFLAGS  := /Ox /EHsc /arch:AVX2
  CXXLDFLAG := /LD legacy_stdio_definitions.lib mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib
  CXXOUTPUT := /Fe

  NVCC	     := nvcc
  NVCCFLAGS  := 
  NVCCLDFLAG := 
  NVCCOUTPUT := 
else
  PYTHON   := python
  EXEEXT   := out
  OBJEXT   := o
  LIBEXT   := a
  SHAREEXT := so

  CXX	    := icc
  CXXFLAGS  := -O3 --std=c++11 -xHost -mkl
  CXXLDFLAG := --shared -fPIC  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx2 -lm
  CXXOUTPUT := -o

  NVCC	     := nvcc
  NVCCFLAGS  := -std=c++11 --compiler-options '-O3 --std=c++11' -arch=sm_60 
  NVCCLDFLAG := --shared --compiler-options '-fPIC' -lcuda -lcudart -lcublas -lcusparse
  NVCCOUTPUT := --output-file #
endif

TARGET := src/pyheomcore.$(SHAREEXT)

SRCDIR  := libsrc
SRC     := $(addprefix $(SRCDIR)/,$(SRC))
SRC_GPU := $(addprefix $(SRCDIR)/,$(SRC_GPU))

all: $(TARGET)

ifeq ($(gpgpu),on)
$(TARGET): $(SRC) $(SRC_GPU)
	$(NVCC) $(NVCCFLAGS) $(SRC) $(SRC_GPU) $(NVCCOUTPUT)$@ $(NVCCLDFLAG)
	-rm -f *.$(OBJEXT) *.$(LIBEXT) *.$(SHAREEXT)
else
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(CXXOUTPUT)$@ $(CXXLDFLAG)
	-rm -f *.$(OBJEXT) *.$(LIBEXT) *.$(SHAREEXT)
endif

install: $(TARGET)
	$(PYTHON) setup.py install # --record files.txt

clean:
	-rm -f $(TARGET) *.$(OBJEXT) *.$(LIBEXT) *.$(SHAREEXT)
	-rm -fr build dist pyheom.egg-info

doc:
	sphinx-apidoc -f -o ./docs ./src
	sphinx-build -b html ./docs ./docs/_build


