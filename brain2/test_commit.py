import brain2
import numpy as np

b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)
b.perceive(b.language.encode("apple"))
b.perceive(b.language.encode("isa"))
b.perceive(b.language.encode("fruit"))
res = b.commit_episode(1.0, b.language.encode("apple"))
print(f"Commit success: {res}")
print(f"Last episode payload length: {len(b.get_last_episode())}")
