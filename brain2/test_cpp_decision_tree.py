#!/usr/bin/env python3
import time
import numpy as np
from engines.synthesis._program_synth_tree import DecisionTree as PyDecisionTree
import brain2

def test_decision_tree():
    print("Generating synthetic data...")
    # Generate random features and labels
    N_SAMPLES = 10000
    N_FEATURES = 20
    N_OPS = 10
    
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = np.random.randint(0, N_OPS, size=N_SAMPLES).astype(np.int32)
    
    # 1. Test Python Implementation
    print("\n--- Python Decision Tree ---")
    py_tree = PyDecisionTree(n_ops=N_OPS, max_depth=10, min_samples=15)
    
    t0 = time.time()
    py_tree.fit(X, y)
    py_fit_time = time.time() - t0
    print(f"Fit Time: {py_fit_time:.4f}s")
    
    t0 = time.time()
    py_preds = [py_tree.predict_dist(x) for x in X[:100]]
    py_pred_time = time.time() - t0
    print(f"Predict Time (100 samples): {py_pred_time:.4f}s")
    
    # 2. Test C++ Implementation
    print("\n--- C++ Decision Tree ---")
    cpp_tree = brain2.DecisionTree(n_ops=N_OPS, max_depth=10, min_samples=15)
    
    t0 = time.time()
    cpp_tree.fit(X, y)
    cpp_fit_time = time.time() - t0
    print(f"Fit Time: {cpp_fit_time:.4f}s")
    
    t0 = time.time()
    cpp_preds = [cpp_tree.predict_dist(x) for x in X[:100]]
    cpp_pred_time = time.time() - t0
    print(f"Predict Time (100 samples): {cpp_pred_time:.4f}s")
    
    # 3. Compare Results
    print("\n--- Comparison ---")
    print(f"Fit Speedup: {py_fit_time / cpp_fit_time:.2f}x")
    print(f"Predict Speedup: {py_pred_time / cpp_pred_time:.2f}x")
    
    differences = 0
    for i in range(100):
        if not np.allclose(py_preds[i], cpp_preds[i], atol=1e-5):
            differences += 1
            if differences == 1:
                print(f"Mismatch at index {i}:")
                print(f"  Py:  {py_preds[i]}")
                print(f"  C++: {cpp_preds[i]}")
    
    if differences == 0:
        print("SUCCESS! C++ and Python outputs match perfectly.")
    else:
        print(f"FAILED: {differences}/100 predictions did not match.")

if __name__ == "__main__":
    test_decision_tree()
