cmake_minimum_required(VERSION 3.14)
project(brain2 VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

# Release flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -ffast-math)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-flto)
endif()

# Python + pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

# OpenMP
if(APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
find_package(OpenMP)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(USE_CUDA)
    set(CUDA_SOURCES core/cuda_math.cu)
    message(STATUS "CUDA: enabled")
endif()

# Brain2 extension module
pybind11_add_module(brain2 brain2.cpp ${CUDA_SOURCES})
target_include_directories(brain2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(brain2 PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(brain2 PRIVATE USE_OPENMP)
    message(STATUS "OpenMP: enabled")
else()
    message(STATUS "OpenMP: not found — single-threaded BMU search")
endif()

# Copy .so to brain2/ root after build so tests can import it directly
add_custom_command(TARGET brain2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:brain2>
            ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:brain2>
    COMMENT "Copying brain2.so to source directory"
)
