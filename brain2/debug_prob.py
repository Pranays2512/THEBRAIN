import brain2
import re

def test_prob(line):
    b = brain2.Brain(8, 8, 16)
    b.symbolic_table.seed_math_symbols()
    for i in range(100): b.symbolic_table.bind(str(i))
    
    parts = line.split("|")
    cat = parts[0].strip()
    q = parts[2].strip()
    exp = parts[3].strip()
    
    words = q.split()
    for w in words: b.language.register_word(w)
    
    prob_match = re.match(r"probability\s*of\s*(\d+)\s*(?:in|out of)\s*(\d+)", q)
    subj_val, obj_val = prob_match.groups()
    b.scratchpad.write("subject", b.language.encode(subj_val), "context")
    b.scratchpad.write("object",  b.language.encode(obj_val),  "context")
    b.scratchpad.write("goal", b.language.encode(cat), "goal")
    seq = b.procedures.retrieve(b.language.encode(cat))
    if seq:
        for op in seq: b.force_reason_step(op, "reply")
        spoken = b.get_spoken_words()
        ans = spoken[-1] if spoken else ""
        if ans.strip() != exp.strip():
            print(f"FAILED: {q} => expected {exp}, got {ans}")
    else:
        print(f"FAILED (No seq): {q}")

with open("tests/test_hardened_1100.txt", "r") as f:
    lines = [x.strip() for x in f if "probability" in x]
    
for l in lines: test_prob(l)
