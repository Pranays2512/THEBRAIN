cmake_minimum_required(VERSION 3.15)
project(brain_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Find Python + pybind11 ────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)
if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Run: pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# ── OpenMP — special handling for Apple Clang ─────────────────────────
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(EXISTS "${LIBOMP_PREFIX}/include/omp.h")
        message(STATUS "Found libomp at ${LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS
            "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        find_package(OpenMP)
    else()
        message(WARNING
            "libomp not found — building without OpenMP (no parallelism).\n"
            "Install with: brew install libomp")
    endif()
else()
    find_package(OpenMP)
endif()

# ── Build the extension ───────────────────────────────────────────────
pybind11_add_module(brain_core  brain_core.cpp)
pybind11_add_module(brain_gru   brain_gru.cpp)
pybind11_add_module(world6_core world6_core.cpp)

# Aggressive optimization
target_compile_options(brain_core  PRIVATE -O3 -ffast-math)
target_compile_options(brain_gru   PRIVATE -O3 -ffast-math)
target_compile_options(world6_core PRIVATE -O3 -ffast-math)

# Apple M-series: tune for ARM
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(brain_core  PRIVATE -mcpu=apple-m1)
    target_compile_options(brain_gru   PRIVATE -mcpu=apple-m1)
    target_compile_options(world6_core PRIVATE -mcpu=apple-m1)
    message(STATUS "Tuning for Apple M-series (arm64)")
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(brain_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(brain_core PRIVATE _OPENMP)
    message(STATUS "OpenMP enabled — parallel BMU search active")
else()
    message(STATUS "OpenMP disabled — single-threaded mode")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
