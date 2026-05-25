import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def test_td_learning():
    print("Testing TD(lambda) Basal Ganglia")
    bg = brain2.BasalGanglia(n_dims=16, lr=0.01)
    
    ctx_a = np.random.randn(16).astype(np.float32)
    goal = np.zeros(16, dtype=np.float32)
    
    success_count = 0
    epochs = 400
    for i in range(epochs):
        # Select operation
        selected_op = bg.select_op(ctx_a, goal, greedy=False)
        
        if selected_op == 5: # RETRIEVE
            reward = 1.0
            success_count += 1
        else:
            reward = -0.1
            
        bg.reinforce(reward)
        
        if (i+1) % 50 == 0:
            print(f"Epoch {i+1}: chosen op {selected_op}, reward {reward}")
            
    print(f"Total successes: {success_count} / {epochs}")
    
    # Test greedy mode
    final_op = bg.select_op(ctx_a, goal, greedy=True)
    print(f"Final selected op: {final_op}")
    
    assert final_op == 5 or success_count > (epochs * 0.1), "TD(lambda) failed to learn the correct operation"
    print("TD(lambda) Actor-Critic test passed.")

if __name__ == '__main__':
    test_td_learning()
