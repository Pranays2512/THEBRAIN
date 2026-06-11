import random
import json
import brain2
import os

def evaluate():
    print("Initializing Brain...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    b.load_components(
        predictor_path="checkpoints/math_brain/predictor.bin",
        language_path="checkpoints/math_brain/language.bin",
        som_path="checkpoints/math_brain/som.bin",
        episodic_path="checkpoints/math_brain/episodic.bin",
        emotion_path="checkpoints/math_brain/emotion.bin",
        self_path="checkpoints/math_brain/self.bin",
        symbolic_path="checkpoints/math_brain/symbolic.bin",
        binding_path="checkpoints/math_brain/binding.bin",
        bg_path="checkpoints/math_brain/bg.bin",
        procedures_path="checkpoints/math_brain/procedures.bin",
        hpred_path="checkpoints/math_brain/hpred.bin"
    )
    b.symbolic_table.seed_math_symbols()
    
    if os.path.exists("checkpoints/semantic_dict.bin"):
        print("Loading GloVe semantic embeddings into Language module...")
        b.language.load_semantics("checkpoints/semantic_dict.bin")
    

    with open("data/math_corpus.json") as f:
        corpus = json.load(f)
        
    test_set = random.sample(corpus, 100)
    
    correct = 0
    for p in test_set:
        inp = p["input"]
        pair = p
        
        words = inp.split()
        b.reset_sequence()
        # We don't need to push to WorkingMemory/SOM since we are only testing the BG reasoning loop directly from the Scratchpad.
        
        # ── Setup the Memory State exactly like training ──
        if inp.startswith("eval"):
            x_val = words[13]
            a_val = words[1]
            b_val = words[5] + words[6]
            c_val = words[8] + words[9]
            
            b.scratchpad.write("subject", b.language.encode(x_val), "math_arg")
            b.scratchpad.write("object", b.language.encode(a_val), "math_arg")
            b.scratchpad.write("a_operator", b.language.encode(b_val), "math_arg")
            b.scratchpad.write("focus", b.language.encode(c_val), "math_arg")
            
            target = pair["target"].replace("is", "").strip()
        elif inp.startswith("roots of"):
            if "=" in words and words.index("=") == 7:
                b_val = "0"
                c_val = words[5] + words[6]
            else:
                b_val = words[5] + words[6]
                c_val = words[8] + words[9]
            
            b.scratchpad.write("object", b.language.encode(b_val), "math_arg")
            b.scratchpad.write("a_operator", b.language.encode(c_val), "math_arg")
            
            target = pair["target"].replace("are", "").strip().replace(" and ", "_and_")
        else:
            if len(words) >= 3:
                b.scratchpad.write("subject", b.language.encode(words[0]), "math_arg")
                b.scratchpad.write("object", b.language.encode(words[2]), "math_arg")
            if len(words) >= 3:
                b.scratchpad.write("a_operator", b.language.encode(words[1]), "math_arg")
                
            target = pair["target"].replace("=", "").strip()
            
        b.start_reasoning()
        sol = b.direct_reason_step("reply")
        
        result_vec = b.scratchpad.read("result")
        if len(result_vec) > 0:
            target_vec = b.language.encode(target)
            dot = sum(a*b for a, b in zip(result_vec, target_vec))
            na = sum(a*a for a in result_vec)
            nb = sum(b*b for b in target_vec)
            if na > 0 and nb > 0:
                sim = dot / ((na**0.5) * (nb**0.5))
                if sim > 0.99:
                    correct += 1
                else:
                    # just print the first 5 dims for debug if it fails
                    print(f"Failed: {inp} | Expected: {target} | Op path: {sol}")
            else:
                print(f"Failed: {inp} | Expected: {target} | Op path: {sol}")
        else:
            print(f"Failed: {inp} | No result from Logic Engine")
            
    print(f"\nAccuracy: {correct}/100")

if __name__ == "__main__":
    evaluate()
