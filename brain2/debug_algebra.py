from tests.run_hardened_suite import load_brain, run_test

b = load_brain()
failures = 0
for i in range(10):
    import random
    a = random.randint(1, 10)
    b_val = random.randint(0, 50)
    c = random.randint(10, 100)
    ans = (c - b_val) / a
    q = f"{a} x + {b_val} = {c}"
    exp = f"x = {ans:.2f}"
    
    a_w, b_w, c_w = str(a), str(b_val), str(c)
    b.reset_sequence()
    b.scratchpad.write("subject",  b.language.encode(c_w), "context")
    b.scratchpad.write("relation", b.language.encode(a_w), "context")
    b.scratchpad.write("object",   b.language.encode(b_w), "context")
    b.force_reason_step(2,  "solve")
    b.force_reason_step(3,  "solve")
    b.force_reason_step(15, "solve")
    spoken = b.get_spoken_words()
    b.clear_spoken_words()
    # Parse float
    ans_raw = spoken[-1] if spoken else "0"
    try:
        ans_str = f"x = {float(ans_raw):.2f}"
    except:
        ans_str = f"x = {ans_raw}"
    print(f"Eq: {q} -> Expected: {exp}, Got: {ans_str}")
