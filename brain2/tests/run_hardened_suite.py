import brain2
import os
import re

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

def run_test(b, category, context, query, expected):
    # Context processing
    if context:
        facts = [x.strip() for x in context.split(";")]
        for fact in facts:
            words = fact.split()
            if category == "causal" and len(words) == 3:
                subj, rel, obj = words
                for w in words:
                    b.language.register_word(w)
                    b.symbolic_table.bind(w)
                b.binding.bind(b.language.encode(subj), b.language.encode(rel), b.language.encode(obj))
            elif category in ["semantic", "grammar", "describe", "self"]:
                subj, rel, obj = words[0], words[1], words[2]
                for w in words:
                    b.language.register_word(w)
                    b.symbolic_table.bind(w)
                b.binding.bind(b.language.encode(subj), b.language.encode(rel), b.language.encode(obj))
            elif category == "episodic":
                # Simulate episodic
                for w in words: b.language.register_word(w)
                subj = b.language.encode(words[0])
                # Emulate Phase 5 Episodic Commit
                b.scratchpad.write("subject", subj, "context")
                b.episodic.observe(b.som.activation_map(subj))
                b.episodic.observe(b.som.activation_map(subj))
                b.commit_episode(1.0, subj)

    # Query processing
    words = query.split()
    for w in words: b.language.register_word(w)
    
    ans = ""
    if category in ["semantic", "grammar", "causal", "self"]:
        # If query starts with "what isa", it's "what isa apple" -> subj=apple, rel=isa
        if words[0] == "what" and words[1] == "isa":
            subj, rel = words[2], words[1]
        elif words[0] == "who" and words[1] == "is":
            subj, rel = words[2], words[1]
        elif len(words) >= 3 and words[2] == "?":
            subj, rel = words[0], words[1]
        else:
            subj, rel = words[0], words[1]
            
        ans_vec, conf = b.binding.query(b.language.encode(subj), b.language.encode(rel), True, 0.3, 4)
        ans = b.language.best_word(ans_vec)
    elif category == "describe":
        subj = words[1]
        properties = b.binding.query_all(b.language.encode(subj), 0.85)
        sentences = []
        for i in range(0, min(20, len(properties)), 2):
            rel_w = b.language.best_word(properties[i])
            obj_w = b.language.best_word(properties[i+1])
            sentences.append(f"{subj} {rel_w} {obj_w}.")
        unique = list(dict.fromkeys(sentences))
        ans = " ".join(unique)
    elif category == "algebra":
        # a x + b = c
        a_w, b_w, c_w = words[0], words[3], words[5]
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
            ans = f"x = {float(ans_raw):.2f}"
        except:
            ans = f"x = {ans_raw}"
    elif category in ["permute", "probability", "area", "power"]:
        perm_match = re.match(r"(\d+)\s*(?:p|permute)\s*(\d+)", query)
        prob_match = re.match(r"probability\s*of\s*(\d+)\s*(?:in|out of)\s*(\d+)", query)
        area_match = re.match(r"area\s*of\s*(\d+)\s*(?:and|by)\s*(\d+)", query)
        pow_match  = re.match(r"(\d+)\s*(?:\^|power)\s*(\d+)", query)
        match = perm_match or prob_match or area_match or pow_match
        if match:
            subj_val, obj_val = match.groups()
            b.reset_sequence()
            b.scratchpad.write("subject", b.language.encode(subj_val), "context")
            b.scratchpad.write("object",  b.language.encode(obj_val),  "context")
            b.scratchpad.write("goal", b.language.encode(category), "goal")
            seq = b.procedures.retrieve(b.language.encode(category))
            if seq:
                for op in seq: b.force_reason_step(op, "reply")
                spoken = b.get_spoken_words()
                b.clear_spoken_words()
                ans = spoken[-1] if spoken else ""
    elif category == "episodic":
        obj = words[-1]
        obj_vec = b.language.encode(obj)
        focus_spike = b.som.activation_map(obj_vec)
        b.scratchpad.write("focus", focus_spike, "curiosity")
        seq = b.procedures.retrieve(b.language.encode("remember"))
        if seq:
            for op in seq: b.force_reason_step(op, "remember")
        spoken = b.get_spoken_words()
        b.clear_spoken_words()
        ans = spoken[-1] if spoken else ""

    return ans.strip() == expected.strip()

def run_suite():
    with open("tests/test_hardened_1100.txt", "r") as f:
        lines = [x.strip() for x in f if x.strip()]

    print(f"Loaded {len(lines)} cases.")
    
    # Phase 1: Individual
    print("\n--- PHASE 1: INDIVIDUAL TESTS ---")
    results_p1 = {c: {"pass": 0, "fail": 0} for c in ["semantic", "describe", "algebra", "permute", "probability", "area", "power", "grammar", "self", "episodic", "causal"]}
    b = load_brain()
    for idx, line in enumerate(lines):
        parts = line.split("|")
        cat = parts[0].strip()
        ctx = parts[1].strip()
        q = parts[2].strip()
        exp = parts[3].strip()
        
        # Phase 1: Clean Brain
        b = load_brain()
        passed = run_test(b, cat, ctx, q, exp)
        if passed:
            results_p1[cat]["pass"] += 1
        else:
            results_p1[cat]["fail"] += 1
        if idx % 100 == 0:
            print(f"Progress Phase 1: {idx}/{len(lines)}")

    print("Phase 1 Results:")
    for k, v in results_p1.items():
        print(f"{k}: {v['pass']}/{v['pass']+v['fail']}")

    # Phase 2: Continuous
    print("\n--- PHASE 2: CONTINUOUS LIFESPAN ---")
    results_p2 = {c: {"pass": 0, "fail": 0} for c in ["semantic", "describe", "algebra", "permute", "probability", "area", "power", "grammar", "self", "episodic", "causal"]}
    b_cont = load_brain()
    for idx, line in enumerate(lines):
        parts = line.split("|")
        cat = parts[0].strip()
        ctx = parts[1].strip()
        q = parts[2].strip()
        exp = parts[3].strip()
        
        passed = run_test(b_cont, cat, ctx, q, exp)
        if passed:
            results_p2[cat]["pass"] += 1
        else:
            results_p2[cat]["fail"] += 1
        if idx % 100 == 0:
            print(f"Progress Phase 2: {idx}/{len(lines)}")
            
    print("Phase 2 Results:")
    for k, v in results_p2.items():
        print(f"{k}: {v['pass']}/{v['pass']+v['fail']}")

if __name__ == "__main__":
    run_suite()
