#  -*- mode:cmake -*-
#  PyHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

cmake_minimum_required(VERSION 3.12)

file(READ "VERSION.txt" VER)
string(STRIP "${VER}" VER)

project(pylibheom)
set(PROJECT_VERSION ${VER})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(exts/libheom)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/exts/libheom/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_BINARY_DIR}/exts/libheom/cmake)

include(libheom-common)
include(libheom-default-compiler-setting)

include_directories(${LIBHEOM_INCLUDES})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  ${LIBHEOM_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${LIBHEOM_CUDA_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS       "${CMAKE_SHARED_LINKER_FLAGS} ${LIBHEOM_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LIBRARY_CUDA_FLAGS "${CMAKE_SHARED_LIBRARY_CUDA_FLAGS} ${LIBHEOM_SHARED_LINKER_FLAGS}")

find_package(PythonInterp 3.6 REQUIRED)

if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/pybind11")
  add_subdirectory(3rdparty/pybind11)
  message(STATUS "Found pybind11 Library in 3rdparty/ directory")
else()
  find_package(pybind11 REQUIRED)
endif()

add_subdirectory(src)


