import brain2
import os

b = brain2.Brain(8, 8, 16)
checkpoint_dir = "checkpoints/stage5_math"
b.load_components(
    predictor_path=os.path.join(checkpoint_dir, "predictor.bin"),
    language_path=os.path.join(checkpoint_dir, "language.bin"),
    som_path=os.path.join(checkpoint_dir, "som.bin"),
    episodic_path=os.path.join(checkpoint_dir, "episodic.bin"),
    emotion_path=os.path.join(checkpoint_dir, "emotion.bin"),
    self_path=os.path.join(checkpoint_dir, "self.bin"),
    symbolic_path=os.path.join(checkpoint_dir, "symbolic.bin"),
    binding_path=os.path.join(checkpoint_dir, "binding.bin"),
    bg_path=os.path.join(checkpoint_dir, "bg.bin"),
    procedures_path=os.path.join(checkpoint_dir, "procedures.bin"),
    hpred_path=os.path.join(checkpoint_dir, "hpred.bin")
)

obj = "obj0"
b.language.register_word(obj)
subj = b.language.encode(obj)
b.scratchpad.write("subject", subj, "context")
b.commit_episode(1.0, subj[:16])

obj_vec = b.language.encode(obj)
focus_spike = b.som.activation_map(obj_vec)
b.scratchpad.write("focus", focus_spike, "curiosity")
seq = b.procedures.retrieve(b.language.encode("remember"))
print(f"Retrieved procedure: {seq}")
if seq:
    for op in seq: b.force_reason_step(op, "remember")
spoken = b.get_spoken_words()
print(f"Spoken: {spoken}")
