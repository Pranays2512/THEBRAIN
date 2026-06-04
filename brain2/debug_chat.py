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

subj_vec = b.language.encode("pranay")
b.commit_episode(1.0, subj_vec)

focus_spike = b.som.activation_map(subj_vec)
b.scratchpad.write("focus", focus_spike, "focus")
b.force_reason_step(6, "goal") # RETRIEVE
res = b.scratchpad.read("result")
print("Retrieved best word:", b.language.best_word(res))
