import brain2
import re

def test_algebra(line):
    b = brain2.Brain(8, 8, 16)
    b.symbolic_table.seed_math_symbols()
    for i in range(100): b.symbolic_table.bind(str(i))
    
    parts = line.split("|")
    cat = parts[0].strip()
    q = parts[2].strip()
    exp = parts[3].strip()
    
    words = q.split()
    for w in words: b.language.register_word(w)
    
    a_w, b_w, c_w = words[0], words[3], words[5]
    b.scratchpad.write("subject",  b.language.encode(c_w), "context")
    b.scratchpad.write("relation", b.language.encode(a_w), "context")
    b.scratchpad.write("object",   b.language.encode(b_w), "context")
    b.force_reason_step(2,  "solve")
    b.force_reason_step(3,  "solve")
    b.force_reason_step(15, "solve")
    spoken = b.get_spoken_words()
    ans_raw = spoken[-1] if spoken else "0"
    try:
        ans = f"x = {float(ans_raw):.2f}"
    except:
        ans = f"x = {ans_raw}"
    
    if ans.strip() != exp.strip():
        print(f"FAILED: {q} => expected {exp}, got {ans}")

with open("tests/test_hardened_1100.txt", "r") as f:
    lines = [x.strip() for x in f if "algebra" in x]
    
for l in lines: test_algebra(l)
