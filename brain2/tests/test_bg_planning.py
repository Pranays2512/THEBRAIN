import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def test_bg_reasoning():
    print("Initializing Brain for BG Planning Test...")
    # Small dimensions for fast testing
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)

    # Register some words so language module can encode/decode
    b.language.register_word("x")
    b.language.register_word("11")
    
    # We want to reason to reach "x"
    print("Testing Brain::reason('x', max_steps=5)...")
    
    # Run the autonomous reasoning loop!
    # This will initialize the scratchpad, use the BG controller to propose operations,
    # evaluate the h_cost of the simulated branches, and traverse the Possibility Tree.
    solution = b.reason("x", max_steps=5)
    
    # Output the selected operations
    op_names = ["READ", "WRITE", "APPLY", "COMPARE", "BIND_QUERY", "RETRIEVE", "ANALOGY", "HALT"]
    
    print("\n--- Solution Path found by BG Controller ---")
    for i, op_idx in enumerate(solution):
        name = op_names[op_idx] if 0 <= op_idx < len(op_names) else f"UNKNOWN({op_idx})"
        print(f"Step {i+1}: {name}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_bg_reasoning()
