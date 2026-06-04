from tests.run_hardened_suite import load_brain, run_test
import brain2

with open("tests/test_hardened_1100.txt", "r") as f:
    lines = [x.strip() for x in f if x.strip()]

b = load_brain()
failures = {}

for idx, line in enumerate(lines):
    parts = line.split("|")
    cat = parts[0].strip()
    ctx = parts[1].strip()
    q = parts[2].strip()
    exp = parts[3].strip()
    
    if cat not in ["semantic", "grammar", "self", "causal", "algebra"]:
        continue
        
    b_test = load_brain()
    passed = run_test(b_test, cat, ctx, q, exp)
    if not passed:
        if cat not in failures:
            failures[cat] = []
        if len(failures[cat]) < 5:
            # Let's re-run it and capture exactly what it spoke
            b_debug = load_brain()
            # reproduce the query processing exactly to see the spoken words
            if ctx:
                facts = [x.strip() for x in ctx.split(";")]
                for fact in facts:
                    words = fact.split()
                    if cat == "causal" and len(words) == 3:
                        subj, rel, obj = words[0], words[1], words[2]
                        for w in words:
                            b_debug.language.register_word(w)
                            b_debug.symbolic_table.bind(w)
                        b_debug.binding.bind(b_debug.language.encode(subj), b_debug.language.encode(rel), b_debug.language.encode(obj))
                    elif cat in ["semantic", "grammar", "describe", "self"]:
                        subj, rel, obj = words[0], words[1], words[2]
                        for w in words:
                            b_debug.language.register_word(w)
                            b_debug.symbolic_table.bind(w)
                        b_debug.binding.bind(b_debug.language.encode(subj), b_debug.language.encode(rel), b_debug.language.encode(obj))
            
            words = q.split()
            for w in words: b_debug.language.register_word(w)
            
            if cat in ["semantic", "grammar", "causal", "self"]:
                if words[0] == "what" and words[1] == "isa":
                    subj, rel = words[2], words[1]
                elif words[0] == "who" and words[1] == "is":
                    subj, rel = words[2], words[1]
                elif len(words) >= 3 and words[2] == "?":
                    subj, rel = words[0], words[1]
                else:
                    subj, rel = words[0], words[1]
                
                b_debug.scratchpad.write("subject", b_debug.language.encode(subj), "curiosity")
                b_debug.scratchpad.write("relation", b_debug.language.encode(rel), "curiosity")
                seq = b_debug.procedures.retrieve(b_debug.language.encode("query"))
                if seq:
                    for op in seq: b_debug.force_reason_step(op, "query")
            
            elif cat == "algebra":
                b_debug.scratchpad.write("subject",  b_debug.language.encode(words[0]), "context")
                b_debug.scratchpad.write("relation", b_debug.language.encode(words[2]), "context")
                b_debug.scratchpad.write("object",   b_debug.language.encode(words[4]), "context")
                b_debug.force_reason_step(2, "solve")
                b_debug.force_reason_step(3, "solve")
                b_debug.force_reason_step(15, "solve")
                
            spoken = b_debug.get_spoken_words()
            ans = spoken[-1] if spoken else ""
            failures[cat].append((ctx, q, exp, ans, spoken))

for cat in failures:
    print(f"\n--- {cat.upper()} FAILURES ---")
    for f in failures[cat]:
        print(f"CTX: {f[0]}\nQ: {f[1]}\nEXP: {f[2]}\nGOT: {f[3]}\nALL SPOKEN: {f[4]}\n")

