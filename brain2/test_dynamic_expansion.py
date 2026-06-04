import sys
import os
import json
import numpy as np

# Adjust path to find the compiled module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def main():
    print("Initializing Brain in 16 dimensions...")
    # brain2 Brain constructor: (som_rows, som_cols, n_dims, hidden_dim, wm_capacity, episodic_max, self_neurons, seed)
    brain = brain2.Brain(5, 5, 16, 64, 5, 200, 16, 42)
    
    # Store some initial state
    print("Writing state in 16D...")
    brain.perceive(np.random.randn(16).astype(np.float32))
    brain.think(2)
    
    print("Before expansion: n_dims =", brain.n_dims)
    assert brain.n_dims == 16, f"Expected 16, got {brain.n_dims}"
    
    print("Expanding to 32 dimensions...")
    brain.expand_dims(32)
    
    print("After expansion: n_dims =", brain.n_dims)
    assert brain.n_dims == 32, f"Expected 32, got {brain.n_dims}"
    
    # Check that perceive and reason still work
    print("Writing state in 32D...")
    brain.perceive(np.random.randn(32).astype(np.float32))
    brain.think(2)
    
    print("Test passed! Dynamic dimensionality expansion is functioning correctly.")

if __name__ == "__main__":
    main()
