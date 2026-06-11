import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "massive_squad")

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

print("Vocab size:", b.language.vocab_size)
print("Knows 'hello':", b.language.knows("hello"))
print("Knows 'tucson':", b.language.knows("tucson"))

b.reset_sequence()
b.perceive(b.language.encode("hello"))
reply = b.think(4)
print("Think from hello:", [w for w in reply.words])
