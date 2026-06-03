import os
import sys
import numpy as np

# Ensure we can import the built brain2 module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def main():
    print("Initializing Brain v3...")
    
    # Use identical configuration as training curriculum
    SOM_ROWS = 8
    SOM_COLS = 8
    N_DIMS = 16
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS)
    
    # Load trained weights
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage5_math")
    if os.path.exists(checkpoint_dir):
        print(f"Loading full Brain architecture from {checkpoint_dir}...")
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
    else:
        print(f"WARNING: Weights not found at {checkpoint_dir}. The brain will guess.")

    # Load identical symbol set AFTER loading components
    b.symbolic_table.seed_math_symbols()
    for i in range(1000):
        b.symbolic_table.bind(str(i))
    words = sorted(["x", "goal", "relation", "math", "op_symbol", "object", "comparison", "eval", "subject", "result", "parse", "reply"])
    for word in words:
        if not b.symbolic_table.knows(word):
            b.language.register_word(word)
            b.symbolic_table.bind(word)
    corpus_words = ["apple", "fruit", "isa", "red", "color", "dog", "animal", "barks", "cat", "meows", "banana", "car", "vehicle", "truck"]
    for word in sorted(corpus_words):
        if not b.symbolic_table.knows(word):
            b.language.register_word(word)
            b.symbolic_table.bind(word)

    print("\n" + "="*50)
    print(" Brain v3 Cognitive Loop is online!")
    print(" It will Perceive your text, Think via PUCT, and Speak.")
    print(" Try asking 'apple isa ?' or 'dog isa ?'")
    print(" (Brain is daydreaming in the background while idle...)")
    print(" Type 'quit' to exit.")
    print("="*50 + "\n")

    import threading
    import time
    import sys
    import random
    
    stop_event = threading.Event()
    def daydream_worker():
        while not stop_event.is_set():
            time.sleep(2.0) # daydream every 2 seconds
            b.daydream()
            
            # Check internal state for spontaneous thought
            # Trigger if highly aroused/surprised, or randomly ~once a minute
            if b.emotion.arousal > 0.5 or random.random() < 0.05:
                topic = ""
                last_payload = b.get_last_episode()
                if len(last_payload) > 0:
                    topic = b.language.best_word(last_payload)
                if not topic or not b.symbolic_table.knows(topic):
                    # Fallback to a random known concept for testing
                    words = ["dog", "sun", "apple", "earth", "bird"]
                    topic = random.choice(words)
                    
                if topic and topic not in ["color", "binding", ""]:
                    topic_vec = b.language.encode(topic)
                    properties = b.binding.query_all(topic_vec, 0.85)
                    
                    sys.stdout.write("\033[2K\r") # Clear the current 'You: ' line
                    if len(properties) >= 2:
                        # Spontaneous Imagination
                        idx = random.choice(range(0, len(properties) - 1, 2))
                        rel_vec = properties[idx]
                        obj_vec = properties[idx+1]
                        rel_word = b.language.best_word(rel_vec)
                        obj_word = b.language.best_word(obj_vec)
                        if not rel_word.isdigit() and not obj_word.isdigit():
                            sys.stdout.write(f"\n\033[96m[🧠 SPONTANEOUS] I just realized that {topic} {rel_word} {obj_word}!\033[0m\n")
                    else:
                        # Curiosity Engine
                        sys.stdout.write(f"\n\033[93m[🧠 CURIOUS] I was just thinking about {topic}... What can you tell me about it?\033[0m\n")
                    
                    sys.stdout.write("You: ")
                    sys.stdout.flush()
            
    daydream_thread = threading.Thread(target=daydream_worker, daemon=True)
    daydream_thread.start()

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                stop_event.set()
                break
                
            import re
            # Add spaces around punctuation
            clean_input = re.sub(r'([?.!])', r' \1 ', user_input)
            words = clean_input.strip().split()
            
            # 1. Phenomenological Identity (Emotion Introspection)
            if user_input.lower().strip(" ?.") in ["how are you", "how do you feel", "what is your state"]:
                v = b.emotion.valence
                a = b.emotion.arousal
                err = b.self_model.mean_recent_error()
                
                # Determine state
                if err > 0.5:
                    mood = "confused"
                elif a > 0.6 and v > 0.2:
                    mood = "excited"
                elif a > 0.6 and v <= 0.2:
                    mood = "stressed"
                elif a <= 0.6 and v > 0.2:
                    mood = "calm and happy"
                else:
                    mood = "calm"
                    
                print(f"Brain: I am {mood}. (valence={v:.2f}, arousal={a:.2f}, pred_error={err:.2f})")
                continue
                
            if len(words) == 2 and words[0] == "describe":
                subj = words[1]
                if not b.symbolic_table.knows(subj):
                    print(f"Brain: I don't know what {subj} is.")
                    continue
                    
                subj_vec = b.language.encode(subj)
                properties = b.binding.query_all(subj_vec, 0.85) # list of [rel, obj, rel, obj...]
                
                if len(properties) == 0:
                    print(f"Brain: I don't know anything about {subj}.")
                    continue
                    
                sentences = []
                # limit to 10 sentences max
                limit = min(20, len(properties)) 
                for i in range(0, limit, 2):
                    rel_vec = properties[i]
                    obj_vec = properties[i+1]
                    rel_word = b.language.best_word(rel_vec)
                    obj_word = b.language.best_word(obj_vec)
                    # skip if we see numbers
                    if obj_word.isdigit() or rel_word.isdigit(): continue
                    sentences.append(f"{subj} {rel_word} {obj_word}.")
                    
                # Deduplicate sentences while preserving order
                unique_sentences = list(dict.fromkeys(sentences))
                print(f"Brain: {' '.join(unique_sentences)}")
                continue

            if len(words) < 3:
                continue
                
            b.scratchpad.clear()
            b.start_reasoning()
            
            op_names = ["READ", "WRITE", "MATH_SUB", "MATH_DIV", "COMPARE", "BIND_QUERY", "RETRIEVE", "ANALOGY", "HALT", "STORE_SUBJ", "STORE_REL", "STORE_OBJ", "NOT"]
            is_query = ("?" in words or "what" in words or "who" in words)
            
            # Neural Parsing phase — write goal then force-route by position (SPO order)
            goal_parse = np.array(b.language.encode("parse"), dtype=np.float32)
            b.scratchpad.write("goal", goal_parse, "goal")
        
            # Check if it's an equation (e.g. 2 x + 4 = 10)
            if "=" in words:
                try:
                    a = words[0]
                    b_val = words[3]
                    c_val = words[5]
                    
                    b.clear_spoken_words()
                    for w in [a, b_val, c_val]:
                        if not b.symbolic_table.knows(w): b.learn_word(w)
                    
                    b.scratchpad.write("subject", b.language.encode(c_val), "context")
                    b.scratchpad.write("relation", b.language.encode(a), "context")
                    b.scratchpad.write("object", b.language.encode(b_val), "context")
                    b.scratchpad.write("goal", b.language.encode("solve"), "goal")
                    
                    b.force_reason_step(2, "solve")  # MATH_SUB
                    b.force_reason_step(3, "solve")  # MATH_DIV
                    b.force_reason_step(15, "solve") # SPEAK
                    b.force_reason_step(8, "solve")  # HALT
                    
                    spoken = b.get_spoken_words()
                    if spoken:
                        print(f"Brain: x = {spoken[-1]}")
                    else:
                        print("Brain: I couldn't solve it.")
                except Exception as e:
                    print("Brain: Math parse error. Use format '2 x + 4 = 10'")
                continue
            
            # 6. Advanced Math (Permutations, Probability, Mensuration, Exponents)
            perm_match = re.match(r"(?:what is )?(\d+)\s*(?:p|permute)\s*(\d+)", user_input.lower())
            prob_match = re.match(r"(?:what is )?(?:the )?probability\s*(?:of )?(\d+)\s*(?:in|out of)\s*(\d+)", user_input.lower())
            area_match = re.match(r"area\s*(?:of )?(?:rectangle )?(\d+)\s*(?:and|by)\s*(\d+)", user_input.lower())
            pow_match = re.match(r"(\d+)\s*(?:\^|power)\s*(\d+)", user_input.lower())

            if perm_match or prob_match or area_match or pow_match:
                try:
                    if perm_match:
                        subj_val, obj_val = perm_match.groups()
                        goal = "permute"
                        prefix = f"Brain: {subj_val}P{obj_val} ="
                    elif prob_match:
                        subj_val, obj_val = prob_match.groups()
                        goal = "probability"
                        prefix = f"Brain: Probability ="
                    elif area_match:
                        subj_val, obj_val = area_match.groups()
                        goal = "area"
                        prefix = f"Brain: Area ="
                    elif pow_match:
                        subj_val, obj_val = pow_match.groups()
                        goal = "power"
                        prefix = f"Brain: {subj_val}^{obj_val} ="

                    # To retrieve from ProceduralMemory, we must provide the same context vector
                    # that was used during consolidation (which is working_mem.context())
                    b.reset_sequence()
                    
                    b.scratchpad.write("subject", b.language.encode(subj_val), "context")
                    b.scratchpad.write("object", b.language.encode(obj_val), "context")
                    goal_vec = b.language.encode(goal)
                    b.scratchpad.write("goal", goal_vec, "goal")
                    # Retrieve procedure: try goal_vec directly first (stable trigger),
                    # fall back to SOM/WM context path
                    seq = b.procedures.retrieve(goal_vec)
                    if not seq:
                        bmu = b.som.activation_map(goal_vec)
                        b.working_mem.gate(bmu * 10.0, 1.0)
                        b.working_mem.tick()
                        ctx = b.working_mem.context()
                        seq = b.procedures.retrieve(ctx)
                    
                    if not seq:
                        print("Brain: I don't know how to compute that.")
                        continue

                    # Execute retrieved procedure
                    for op in seq:
                        b.force_reason_step(op, goal)
                    
                    spoken = b.get_spoken_words()
                    if spoken:
                        print(f"{prefix} {spoken[-1]}")
                    else:
                        print("Brain: I couldn't compute the answer.")
                except Exception as e:
                    pass
                continue

            parse_words = [w.strip("?") for w in words if w not in ["?", "a", "an", "the"]]
            parse_words = ["isa" if w == "is" else w for w in parse_words]
            
            # If word is "not", remember it to negate the object later
            slot_ops = [9, 10, 11]  # STORE_SUBJ, STORE_REL, STORE_OBJ
            slot_idx = 0
            has_not = False
            curiosity_triggered = None
            
            for i, w in enumerate(parse_words):
                if w == "not":
                    has_not = True
                    continue
                    
                is_novel = not b.symbolic_table.knows(w)
                if is_novel:
                    b.learn_word(w)
                    # Phase 6: Curiosity & Active Learning
                    curiosity_triggered = w
                    
                vec = b.language.encode(w)
                if sum(abs(x) for x in vec) > 0:
                    res = b.perceive(vec)
                    
                    if is_novel:
                        b.force_reason_step(14, "curiosity") # ASK_USER
                    
                    op = slot_ops[min(slot_idx, 2)]
                    b.force_reason_step(op, "parse")
                    slot_idx += 1
            
            if has_not:
                b.force_reason_step(12, "parse") # NOT
                    
            # Reply phase
            if curiosity_triggered:
                reply = f"What is {curiosity_triggered}?"
            elif "say" in words or "remember" in words or "history" in words:
                last_payload = b.get_last_episode()
                if len(last_payload) > 0:
                    topic = b.language.best_word(last_payload)
                    reply = f"(retrieving from episodic memory...) you were talking about '{topic}'."
                else:
                    reply = "I don't remember anything yet."
            else:
                # Full autonomy
                b.clear_spoken_words()
                b.scratchpad.write("goal", b.language.encode("reply"), "goal")
                
                if is_query:
                    b.force_reason_step(5, "reply")  # BIND_QUERY
                    b.force_reason_step(17, "reply") # SPEAK_SUBJ
                    b.force_reason_step(18, "reply") # SPEAK_REL
                    b.force_reason_step(15, "reply") # SPEAK (result)
                    b.force_reason_step(8, "reply")  # HALT
                    spoken = b.get_spoken_words()
                    if spoken:
                        reply = " ".join(spoken) + "."
                        if has_not: reply = "not " + reply
                    else:
                        reply = "I don't know."
                else:
                    subj_vec = b.scratchpad.read("subject")
                    rel_vec = b.scratchpad.read("relation")
                    obj_vec = b.scratchpad.read("object")
                    if len(subj_vec) > 0 and len(rel_vec) > 0 and len(obj_vec) > 0:
                        b.binding.bind(subj_vec, rel_vec, obj_vec)
                    b.force_reason_step(8, "reply")  # HALT
                    reply = "Got it."
                        
            if reply in ["color", "binding", ""]:
                reply = "..."
                    
            print(f"Brain: {reply}")
            
            # Phase 5: Commit the conversation turn into Episodic Memory
            subj_vec = b.scratchpad.read("subject")
            if len(subj_vec) > 0 and sum(abs(x) for x in subj_vec) > 1e-6:
                b.commit_episode(1.0, subj_vec[:16])
            
        except (KeyboardInterrupt, EOFError):
            break
            
    print("\nSaving Brain state to disk...")
    if os.path.exists(checkpoint_dir):
        b.save_components(checkpoint_dir)
    print("Shutting down Brain.")

if __name__ == "__main__":
    main()
