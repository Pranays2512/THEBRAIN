import os, sys, brain2
N_DIMS = 128
SOM_ROWS = 256
SOM_COLS = 256
HIDDEN_DIM = 256
b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
ckpt_dir = "checkpoints/fluent_brain"
b.load_components(
    predictor_path=os.path.join(ckpt_dir, "predictor.bin"),
    language_path=os.path.join(ckpt_dir, "language.bin"),
    som_path=os.path.join(ckpt_dir, "som.bin"),
    episodic_path=os.path.join(ckpt_dir, "episodic.bin"),
    emotion_path=os.path.join(ckpt_dir, "emotion.bin"),
    self_path=os.path.join(ckpt_dir, "self.bin"),
    symbolic_path=os.path.join(ckpt_dir, "symbolic.bin"),
    binding_path=os.path.join(ckpt_dir, "binding.bin"),
    bg_path=os.path.join(ckpt_dir, "bg.bin"),
    procedures_path=os.path.join(ckpt_dir, "procedures.bin"),
    hpred_path=os.path.join(ckpt_dir, "hpred.bin")
)
test_prompts = ["what is your name", "what do you think of space", "how are you"]
for p in test_prompts:
    b.reset_sequence()
    for w in p.split():
        if b.language.knows(w):
            b.perceive(b.language.encode(w))
    res = b.think(4)
    print(f"[{p}] -> {' '.join([w for w in res.words if w])}")
