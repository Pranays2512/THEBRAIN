"""
run_tests.py — Batch test runner for brain2
Covers: semantic_query, describe, algebra, permute, probability, area, power
"""
import brain2
import os
import re
import numpy as np

# ── Load Brain ────────────────────────────────────────────────────────────────
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
b.symbolic_table.seed_math_symbols()
for i in range(1000):
    b.symbolic_table.bind(str(i))

print("Brain loaded. Running tests...")

# ── Read test cases ───────────────────────────────────────────────────────────
with open("test_700.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

results = []

def vec_norm(v):
    """Safe L1 norm of a list/numpy vector."""
    return float(np.sum(np.abs(np.asarray(v, dtype=np.float32))))

def run_semantic_query(words):
    """dog isa ? → looks up via 'reply' procedure"""
    subj = words[0]
    rel  = words[1]
    
    b.reset_sequence()
    b.scratchpad.write("subject", b.language.encode(subj), "context")
    b.scratchpad.write("relation", b.language.encode(rel), "context")
    b.scratchpad.write("object", b.language.encode("?"), "context")
    goal_vec = b.language.encode("reply")
    b.scratchpad.write("goal", goal_vec, "goal")

    seq = b.procedures.retrieve(goal_vec)
    if not seq:
        bmu = b.som.activation_map(goal_vec)
        b.working_mem.gate(bmu * 10.0, 1.0)
        b.working_mem.tick()
        ctx = b.working_mem.context()
        seq = b.procedures.retrieve(ctx)

    if not seq:
        return "I don't know."

    for op in seq:
        b.force_reason_step(op, "reply")

    spoken = b.get_spoken_words()
    b.clear_spoken_words()
    # If the output is "dog isa animal", we just return the final word for the test
    if len(spoken) >= 3:
        return spoken[-1]
    elif len(spoken) == 1:
        return spoken[0]
    return "I don't know."

def run_describe(words):
    """describe dog → list known relations"""
    subj = words[1] if len(words) > 1 else words[0]
    subj_vec    = b.language.encode(subj)
    properties  = b.binding.query_all(subj_vec, 0.85)
    if not properties:
        return "I don't know anything about that."
    sentences = []
    for i in range(0, min(20, len(properties)), 2):
        rel_w = b.language.best_word(properties[i])
        obj_w = b.language.best_word(properties[i+1])
        if not obj_w.isdigit() and not rel_w.isdigit():
            sentences.append(f"{subj} {rel_w} {obj_w}.")
    unique = list(dict.fromkeys(sentences))
    return " ".join(unique) if unique else "I don't know anything about that."

def run_algebra(words):
    """a x + b = c  →  x = (c-b)/a"""
    try:
        a_w, b_w, c_w = words[0], words[3], words[5]
        for w in [a_w, b_w, c_w]:
            if not b.symbolic_table.knows(w):
                b.learn_word(w)
        b.reset_sequence()
        b.scratchpad.write("subject",  b.language.encode(c_w), "context")
        b.scratchpad.write("relation", b.language.encode(a_w), "context")
        b.scratchpad.write("object",   b.language.encode(b_w), "context")
        b.scratchpad.write("goal",     b.language.encode("solve"), "goal")
        b.force_reason_step(2,  "solve")   # MATH_SUB
        b.force_reason_step(3,  "solve")   # MATH_DIV
        b.force_reason_step(15, "solve")   # SPEAK
        b.force_reason_step(8,  "solve")   # HALT
        spoken = b.get_spoken_words()
        b.clear_spoken_words()
        return f"x = {spoken[-1]}" if spoken else "I couldn't solve it."
    except Exception:
        return "Math parse error."

def run_procedural(user_input):
    """permute / probability / area / power via ProceduralMemory"""
    perm_match = re.match(r"(\d+)\s*(?:p|permute)\s*(\d+)", user_input.lower())
    prob_match = re.match(r"probability\s*of\s*(\d+)\s*(?:in|out of)\s*(\d+)", user_input.lower())
    area_match = re.match(r"area\s*of\s*(\d+)\s*(?:and|by)\s*(\d+)", user_input.lower())
    pow_match  = re.match(r"(\d+)\s*(?:\^|power)\s*(\d+)", user_input.lower())

    match = perm_match or prob_match or area_match or pow_match
    if not match:
        return "Math parse error."

    subj_val, obj_val = match.groups()
    if perm_match:   goal = "permute"
    elif prob_match: goal = "probability"
    elif area_match: goal = "area"
    else:            goal = "power"

    try:
        b.reset_sequence()
        b.scratchpad.write("subject", b.language.encode(subj_val), "context")
        b.scratchpad.write("object",  b.language.encode(obj_val),  "context")
        goal_vec = b.language.encode(goal)
        b.scratchpad.write("goal", goal_vec, "goal")

        # Use goal_vec directly as retrieval key (matches new consolidate trigger)
        seq = b.procedures.retrieve(goal_vec)
        # Fallback: SOM/WM context path
        if not seq:
            bmu = b.som.activation_map(goal_vec)
            b.working_mem.gate(bmu * 10.0, 1.0)
            b.working_mem.tick()
            ctx = b.working_mem.context()
            seq = b.procedures.retrieve(ctx)

        if not seq:
            return "I don't know how to compute that."

        for op in seq:
            b.force_reason_step(op, goal)

        spoken = b.get_spoken_words()
        b.clear_spoken_words()
        return str(spoken[-1]) if spoken else "I couldn't solve it."
    except Exception:
        return "Math parse error."


# ── Main loop ─────────────────────────────────────────────────────────────────
feature_counts = {}

for idx, line in enumerate(lines):
    if not line.startswith("["):
        continue

    bracket_end  = line.index("]")
    feature_tag  = line[1:bracket_end]
    user_input   = line[bracket_end+2:].strip()
    words        = user_input.lower().replace("?","").replace(".","").split()

    if not words:
        continue

    feature_counts[feature_tag] = feature_counts.get(feature_tag, 0) + 1
    b.clear_spoken_words()

    # Route to correct handler
    if feature_tag == "describe":
        output = run_describe(words)

    elif feature_tag == "algebra":
        output = run_algebra(words)

    elif feature_tag in ("permute", "probability", "area", "power"):
        output = run_procedural(user_input)

    else:  # semantic_query
        output = run_semantic_query(words)

    result_line = f"[{feature_tag}] Q: {user_input} | A: {output}"
    results.append(result_line)

    # Print progress every 50 cases
    if (idx + 1) % 50 == 0:
        print(f"  [{idx+1}/700] done...")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("test_results_700.txt", "w") as f:
    for res in results:
        f.write(res + "\n")

print(f"\nDone! {len(results)} results saved to test_results_700.txt")
print("Feature counts:", feature_counts)
