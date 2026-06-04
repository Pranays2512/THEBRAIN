from tests.run_hardened_suite import load_brain
import brain2

b = load_brain()

words = ["obj0"]
for w in words: b.language.register_word(w)
subj = b.language.encode(words[0])
b.scratchpad.write("subject", subj, "context")

b.episodic.observe(b.som.activation_map(subj))
b.episodic.observe(b.som.activation_map(subj))
success = b.commit_episode(1.0, subj)
print(f"Commit success: {success}")

words = ["obj0"]
for w in words: b.language.register_word(w)

obj = words[-1]
obj_vec = b.language.encode(obj)
focus_spike = b.som.activation_map(obj_vec)
b.scratchpad.write("focus", focus_spike, "curiosity")
seq = b.procedures.retrieve(b.language.encode("remember"))
if seq:
    for op in seq:
        b.force_reason_step(op, "remember")
print("Spoken words:", b.get_spoken_words())

