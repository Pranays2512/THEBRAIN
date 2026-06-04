import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import brain2

b = brain2.Brain(som_rows=4, som_cols=4, n_dims=16)

vec = np.ones(16, dtype=np.float32) * 5.0
print("Before gate:")
print(b.working_mem.activations())
b.working_mem.gate(vec, 1.0)
print("After gate:")
print(b.working_mem.activations())
b.think()
print("After think:")
print(b.working_mem.activations())
ctx = b.working_mem.context()
print("Context:", ctx)

