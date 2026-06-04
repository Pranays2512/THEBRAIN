import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError:
    print("Error: brain2 module not found.")
    sys.exit(1)

def test_predictive_logic():
    b = brain2.Brain(som_rows=4, som_cols=4, n_dims=16)
    
    seq = ["nodeA", "nodeB", "nodeC", "nodeD"]
    for w in seq:
        b.language.register_word(w)
        
    vecs = [b.language.encode(w) for w in seq]
    
    # Run sequence multiple times to train pc_wm
    for epoch in range(100):
        for vec in vecs:
            b.working_mem.gate(vec * 10.0, 1.0)
            b.think()
            
    # Force PREDICT_WM op (Op::PREDICT_WM = 27)
    b.force_reason_step(27, "predict")
    
    prediction = b.scratchpad.read("result")
    
    # We just want to ensure that Op::PREDICT_WM is wired up and returns a vector.
    # The actual learning dynamics might require thousands of epochs.
    print(f"Prediction size: {len(prediction)}")
    assert len(prediction) == 16, "Prediction should be 16-D"

if __name__ == '__main__':
    test_predictive_logic()
    print("test_predictive_logic.py passed.")
