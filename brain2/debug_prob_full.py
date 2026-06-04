import brain2
import os, re

def load_brain():
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
    return b

def test_prob(b, line):
    parts = line.split("|")
    cat = parts[0].strip()
    q = parts[2].strip()
    exp = parts[3].strip()
    
    words = q.split()
    for w in words: b.language.register_word(w)
    
    prob_match = re.match(r"probability\s*of\s*(\d+)\s*(?:in|out of)\s*(\d+)", q)
    if not prob_match:
        print(f"NO MATCH: {q}")
        return
    subj_val, obj_val = prob_match.groups()
    b.reset_sequence()
    b.scratchpad.write("subject", b.language.encode(subj_val), "context")
    b.scratchpad.write("object",  b.language.encode(obj_val),  "context")
    b.scratchpad.write("goal", b.language.encode(cat), "goal")
    seq = b.procedures.retrieve(b.language.encode(cat))
    if seq:
        for op in seq: b.force_reason_step(op, "reply")
        spoken = b.get_spoken_words()
        b.clear_spoken_words()
        ans = spoken[-1] if spoken else ""
        if ans.strip() != exp.strip():
            print(f"FAILED: {q} => expected {exp}, got {ans}")
    else:
        print(f"FAILED (No seq): {q}")

with open("tests/test_hardened_1100.txt", "r") as f:
    lines = [x.strip() for x in f if "probability" in x]

failures = 0
b = load_brain()
for l in lines:
    test_prob(b, l)

