# -*- mode:cmake -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

if (LIBHEOM_ENABLE_CUDA_INTERNAL)
  process_jinja2(pylibheom.cu SOURCE pylibheom.cc.j2)
  pybind11_add_module(pylibheom pylibheom.cu)
  set_target_properties(pylibheom PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(pylibheom PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
else()
  process_jinja2(pylibheom.cc)
  pybind11_add_module(pylibheom pylibheom.cc)
endif()
target_link_libraries(pylibheom PRIVATE libheom)
