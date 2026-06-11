import time
import brain2

b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
t0 = time.time()
b.perceive_text("what is the capital of france")
t1 = time.time()
print(f"Perceive time: {t1-t0:.4f}s")
print(b.get_profiling_report())
