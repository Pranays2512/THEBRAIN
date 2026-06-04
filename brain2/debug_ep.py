import brain2
b = brain2.Brain(8, 8, 16)
ckpt = "checkpoints/stage5_math"
b.load_components(
    predictor_path=f"{ckpt}/predictor.bin",
    language_path=f"{ckpt}/language.bin",
    som_path=f"{ckpt}/som.bin",
    episodic_path=f"{ckpt}/episodic.bin",
    emotion_path=f"{ckpt}/emotion.bin",
    self_path=f"{ckpt}/self.bin",
    symbolic_path=f"{ckpt}/symbolic.bin",
    binding_path=f"{ckpt}/binding.bin",
    bg_path=f"{ckpt}/bg.bin",
    procedures_path=f"{ckpt}/procedures.bin",
    hpred_path=f"{ckpt}/hpred.bin"
)

# Look at episodic memory directly
v_pranay = b.language.encode("pranay")
sp = b.som.activation_map(v_pranay)
topk = b.episodic.retrieve_topk(sp, 5)
print("Top 5 matches for 'pranay':")
for sim, idx in topk:
    ep = b.episodic.get_episode(idx)
    if ep:
        print(f"Idx: {idx}, Sim: {sim}, Payload: {b.language.best_word(ep.payload)}")
