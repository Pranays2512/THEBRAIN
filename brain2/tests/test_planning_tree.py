import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def test_possibility_tree():
    print("Testing Scratchpad Possibility Tree...")
    n_dims = 16
    scratchpad = brain2.Scratchpad(n_dims)

    # 1. Initialize root state
    # e.g., representing "2x + 3 = 11"
    root_state = np.random.randn(n_dims).astype(np.float32)
    root_id = scratchpad.start_tree(root_state)
    print(f"Root Node ID: {root_id}")

    # 2. Create branches with different heuristic costs (h_cost)
    # Lower h_cost means closer to goal
    
    # Branch A: "Add 7" -> bad move, high h_cost
    state_a = np.random.randn(n_dims).astype(np.float32)
    id_a = scratchpad.branch(state_a, 0.9)
    print(f"Branch A created. ID: {id_a}, h_cost: 0.9")

    # Branch B: "Subtract 3" -> great move, low h_cost
    state_b = np.random.randn(n_dims).astype(np.float32)
    id_b = scratchpad.branch(state_b, 0.3)
    print(f"Branch B created. ID: {id_b}, h_cost: 0.3")

    # Branch C: "Divide by 2" -> impossible right now, medium h_cost
    state_c = np.random.randn(n_dims).astype(np.float32)
    id_c = scratchpad.branch(state_c, 0.7)
    print(f"Branch C created. ID: {id_c}, h_cost: 0.7")

    # 3. Ask Scratchpad to move to the best child
    best_id = scratchpad.move_to_best_child()
    print(f"\nMoved to best child. ID selected: {best_id}")

    assert best_id == id_b, "Expected to move to Branch B (lowest h_cost)!"
    print("SUCCESS: Selected the branch with the lowest heuristic cost!")

    # 4. Verify current state is Branch B's state
    current_state = scratchpad.current_tree_state()
    assert np.allclose(current_state, state_b), "Current state does not match Branch B's state!"
    print("SUCCESS: Current tree state matches Branch B state.")

    # 5. Expand from Branch B
    # Branch D: "Divide by 2" -> Goal!
    state_d = np.random.randn(n_dims).astype(np.float32)
    id_d = scratchpad.branch(state_d, 0.0)
    print(f"\nBranch D created from {best_id}. ID: {id_d}, h_cost: 0.0")

    best_id = scratchpad.move_to_best_child()
    print(f"Moved to best child. ID selected: {best_id}")
    assert best_id == id_d, "Expected to move to Branch D!"
    print("SUCCESS: Goal state reached.")

    # 6. Test backtracking
    print("\nBacktracking to root...")
    scratchpad.move_to(root_id)
    assert scratchpad.current_node() == root_id, "Failed to backtrack to root"
    print("SUCCESS: Backtracked successfully.")

    scratchpad.clear_tree()
    print("Cleared Tree.")
    print("\nAll Planning Tree tests passed successfully!")

if __name__ == "__main__":
    test_possibility_tree()
