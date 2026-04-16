#!/bin/bash
# build.sh — Build brain_core C++ extension
# Run once. After this, brain_fast.py imports it automatically.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_cpp"

echo "=== Building brain_core C++ extension ==="
echo ""

# Check dependencies
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install with: brew install cmake"
    exit 1
fi

if ! python3 -c "import pybind11" &>/dev/null; then
    echo "ERROR: pybind11 not found. Install with: pip install pybind11"
    exit 1
fi

# Check libomp
LIBOMP="$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib"
if [ ! -f "$LIBOMP" ]; then
    echo "WARNING: libomp not found — building without OpenMP (no parallelism)"
    echo "         Install with: brew install libomp"
    echo ""
fi

# Configure
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    2>&1

# Build
echo ""
echo "=== Compiling ==="
cmake --build . --config Release -j"$(sysctl -n hw.ncpu)" 2>&1

# Copy .so to project root
SO_FILE=$(find . -name "brain_core*.so" | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: build produced no .so file"
    exit 1
fi

cp "$SO_FILE" "$SCRIPT_DIR/"
echo ""
echo "=== Done ==="
echo ""

# Verify
cd "$SCRIPT_DIR"
python3 -c "
import brain_core
print(f'  brain_core loaded successfully')
print(f'  OpenMP: {brain_core.has_openmp}')
print(f'  Threads: {brain_core.n_threads}')

# Quick test
import numpy as np
som = brain_core.FastSOM(rows=10, cols=10, n_dims=13)
vec = np.random.randn(13).astype(np.float32)
bmu = som.find_bmu(vec)
som.update(vec, bmu, 1.0)
print(f'  SOM test: 100 neurons, BMU={bmu} ✓')

tp = brain_core.FastTP(n_neurons=100)
tp.observe(bmu, (bmu+1) % 100)
dist = tp.get_distribution([bmu])
next_bmu = tp.sample(dist)
print(f'  TP test: transition {bmu}→{next_bmu} ✓')
print()
print('Ready. Run: python train_fast.py')
"
