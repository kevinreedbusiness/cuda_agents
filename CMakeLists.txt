cmake_minimum_required(VERSION 3.18)
project(ParallelOptimizationEngine LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_PYBIND "Build Python bindings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(poe_engine STATIC src/engine.cpp src/hardware.cpp)
target_include_directories(poe_engine PUBLIC include)
target_compile_definitions(poe_engine
    PUBLIC
        $<$<BOOL:${ENABLE_CUDA}>:POE_ENABLE_CUDA>
        $<$<BOOL:${ENABLE_PYBIND}>:POE_ENABLE_PYBIND>)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(poe_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(poe_cuda STATIC src/cuda_backend.cu)
    target_include_directories(poe_cuda PUBLIC include)
    target_link_libraries(poe_cuda PUBLIC CUDA::cudart)
    target_compile_definitions(poe_cuda PUBLIC POE_ENABLE_CUDA)
    target_link_libraries(poe_engine PUBLIC poe_cuda)
endif()

add_executable(poe_cli src/main.cpp)
target_link_libraries(poe_cli PRIVATE poe_engine)

if(ENABLE_PYBIND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(poe_engine_module src/bindings.cpp)
    target_link_libraries(poe_engine_module PRIVATE poe_engine)
    target_compile_definitions(poe_engine_module PRIVATE POE_ENABLE_PYBIND)
    set_target_properties(poe_engine_module PROPERTIES OUTPUT_NAME poe_engine)
endif()

install(TARGETS poe_cli poe_engine
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)

