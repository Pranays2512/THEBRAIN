import os, sys, random, time
import numpy as np
import brain2

OP_BIND_QUERY = 5
OP_HALT = 8
OP_SPEAK = 15
OP_SPEAK_SUBJ = 17
OP_SPEAK_REL = 18

b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)

ckpt_dir = "checkpoints/stage5_math"
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

# CONSOLIDATE THE PROCEDURE
print("Consolidating 'reply' procedure to Procedural Memory...")
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("reply"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(grammar_ops, "reply")

print("Testing Grammar sequence via Procedural Memory...")
b.scratchpad.clear()
b.scratchpad.write("subject", b.language.encode("apple"), "context")
b.scratchpad.write("relation", b.language.encode("isa"), "context")
b.scratchpad.write("object", b.language.encode("?"), "context")
goal_vec = b.language.encode("reply")
b.scratchpad.write("goal", goal_vec, "goal")

b.clear_spoken_words()
seq = b.procedures.retrieve(goal_vec)
if not seq:
    bmu = b.som.activation_map(goal_vec)
    b.working_mem.gate(bmu * 10.0, 1.0)
    b.working_mem.tick()
    ctx = b.working_mem.context()
    seq = b.procedures.retrieve(ctx)

if seq:
    print("Successfully retrieved procedure from neural memory.")
    for op in seq:
        b.force_reason_step(op, "reply")
else:
    print("Failed to retrieve procedure.")

print("Spoken words:", b.get_spoken_words())

b.save_components(ckpt_dir)
print("Done!")
