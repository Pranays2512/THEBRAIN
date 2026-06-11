import brain2
b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
b.load_components(language_path="checkpoints/math_brain/language.bin", predictor_path="checkpoints/math_brain/predictor.bin", som_path="checkpoints/math_brain/som.bin", episodic_path="checkpoints/math_brain/episodic.bin", emotion_path="checkpoints/math_brain/emotion.bin", self_path="checkpoints/math_brain/self.bin", symbolic_path="checkpoints/math_brain/symbolic.bin")
p = b.language.encode("+")
m = b.language.encode("-")
print("+ == - ?", p == m)
import math
dot = sum(x*y for x, y in zip(p, m))
print("Dot product:", dot)
