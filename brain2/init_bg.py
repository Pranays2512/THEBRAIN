import brain2
import json

b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
b.save_components("checkpoints/math_brain")
