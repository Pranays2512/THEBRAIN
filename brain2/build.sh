#!/bin/bash
set -e
cd "$(dirname "$0")"

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

PYBIND11_DIR=$(../../venv/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

make -j$(sysctl -n hw.logicalcpu)

echo ""
echo "Build complete. brain2.so in brain2/"
