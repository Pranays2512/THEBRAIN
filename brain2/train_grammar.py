import os, sys, random, time
import numpy as np
import brain2

OP_BIND_QUERY = 5
OP_HALT = 8
OP_SPEAK = 15
OP_SPEAK_SUBJ = 17
OP_SPEAK_REL = 18

b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)

ckpt_dir = "checkpoints/stage4_parsing"
try:
    b.load_components(
        predictor_path=f"{ckpt_dir}/predictor.bin",
        language_path=f"{ckpt_dir}/language.bin",
        som_path=f"{ckpt_dir}/som.bin",
        episodic_path=f"{ckpt_dir}/episodic.bin",
        emotion_path=f"{ckpt_dir}/emotion.bin",
        self_path=f"{ckpt_dir}/self.bin",
        symbolic_path=f"{ckpt_dir}/symbolic.bin",
        binding_path=f"{ckpt_dir}/binding.bin",
        bg_path=f"{ckpt_dir}/bg.bin",
        procedures_path=f"{ckpt_dir}/procedures.bin",
        hpred_path=f"{ckpt_dir}/hpred.bin"
    )
except Exception as e:
    print("Failed to load components:", e)
    sys.exit(1)

grammar_ops = [OP_BIND_QUERY, OP_SPEAK_SUBJ, OP_SPEAK_REL, OP_SPEAK, OP_HALT]
epochs = 1000

print(f"Training Grammar Engine (Teacher Forcing)")
for epoch in range(epochs):
    b.scratchpad.clear()
    b.clear_spoken_words()
    
    for w in ["apple", "isa", "?", "fruit", "reply"]:
        if not b.symbolic_table.knows(w):
            b.learn_word(w)
            
    # Fake query: apple isa ?
    b.scratchpad.write("subject", b.language.encode("apple"), "context")
    b.scratchpad.write("relation", b.language.encode("isa"), "context")
    b.scratchpad.write("object", b.language.encode("?"), "context")
    b.scratchpad.write("goal", b.language.encode("reply"), "goal")

    b.start_reasoning()
    for op in grammar_ops:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(1.0)

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{epochs} complete.")

print("Testing Grammar sequence...")
b.scratchpad.clear()
b.scratchpad.write("subject", b.language.encode("apple"), "context")
b.scratchpad.write("relation", b.language.encode("isa"), "context")
b.scratchpad.write("object", b.language.encode("?"), "context")
b.scratchpad.write("goal", b.language.encode("reply"), "goal")

b.start_reasoning()
b.clear_spoken_words()
ops_taken = []
for step in range(10):
    op = b.reason_step("reply", 0.0) # Greedy inference
    ops_taken.append(op)
    if op == OP_HALT:
        break

print("Ops taken:", ops_taken)
print("Spoken words:", b.get_spoken_words())

b.save_components(ckpt_dir)
print("Done!")
