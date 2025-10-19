cmake_minimum_required(VERSION 3.20)
project(ParallelOptimizationEngine LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA kernels" ON)
option(ENABLE_PYBIND "Enable Python bindings via pybind11" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: enabling")
  set(HAS_OPENMP 1)
else()
  message(STATUS "OpenMP not found: building without OpenMP")
  set(HAS_OPENMP 0)
endif()

# CUDA (optional)
if(ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: enabling GPU support")
    set(HAS_CUDA 1)
  else()
    message(WARNING "CUDA Toolkit not found; building without CUDA")
    set(HAS_CUDA 0)
  endif()
else()
  set(HAS_CUDA 0)
endif()

# pybind11 (optional)
if(ENABLE_PYBIND)
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    message(STATUS "pybind11 found: enabling Python bindings")
    set(HAS_PYBIND 1)
  else()
    message(WARNING "pybind11 not found; Python bindings disabled")
    set(HAS_PYBIND 0)
  endif()
else()
  set(HAS_PYBIND 0)
endif()

# Configure definitions
add_compile_definitions(HAS_OPENMP=${HAS_OPENMP} HAS_CUDA=${HAS_CUDA})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

if(HAS_PYBIND)
  add_subdirectory(python)
endif()
