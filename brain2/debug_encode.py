import brain2
import time

b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
b.load_components(
    predictor_path="checkpoints/math_brain/predictor.bin",
    language_path="checkpoints/math_brain/language.bin",
    som_path="checkpoints/math_brain/som.bin",
    episodic_path="checkpoints/math_brain/episodic.bin",
    emotion_path="checkpoints/math_brain/emotion.bin",
    self_path="checkpoints/math_brain/self.bin",
    symbolic_path="checkpoints/math_brain/symbolic.bin",
    binding_path="checkpoints/math_brain/binding.bin",
    bg_path="checkpoints/math_brain/bg.bin",
    procedures_path="checkpoints/math_brain/procedures.bin",
    hpred_path="checkpoints/math_brain/hpred.bin"
)

import uuid
for i in range(10):
    t0 = time.time()
    w = str(uuid.uuid4())
    b.language.encode(w)
    t1 = time.time()
    print(f"encode new {i}: {t1-t0:.6f}")
