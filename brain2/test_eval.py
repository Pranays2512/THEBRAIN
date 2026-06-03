import brain2
import numpy as np

b = brain2.Brain(som_rows=10, som_cols=10, n_dims=16, episodic_max=10000)
b.symbolic_table.seed_math_symbols()
for i in range(101):
    b.symbolic_table.bind(str(i))
b.language.register_word("x")

b.load_bg("checkpoints/bg_ep50000.bin")

def encode(v): return b.symbolic_table.lookup(str(v))

b.scratchpad.clear()
b.scratchpad.write("subject",  np.array(encode(10), dtype=np.float32), "math")
b.scratchpad.write("object",   np.array(encode(4), dtype=np.float32), "math")
b.scratchpad.write("relation", np.array(encode(2), dtype=np.float32), "math")
b.scratchpad.write("goal",     np.array(b.language.encode("x"), dtype=np.float32), "goal")
b.scratchpad.write("op_symbol", np.array(encode(0), dtype=np.float32), "-")
b.scratchpad.write("comparison", np.zeros(16, dtype=np.float32), "eval")

b.start_reasoning()
for _ in range(6):
    op = b.reason_step("x", 0.0)
    print("Picked:", op)
    if op == 8: break
