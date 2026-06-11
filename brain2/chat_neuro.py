#!/usr/bin/env python3
"""
chat_neuro.py — Real-Time Chat with the Fully Integrated Neuro-Symbolic Brain

This script acts as the Brain's Prefrontal Cortex NLP Router. 
It intercepts human speech, detects logical structure, populates the Brain's 
Scratchpad, and consciously triggers the Basal Ganglia reasoning loop.
"""

import os, sys, re, shutil, datetime, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def parse_nlp(user_input):
    """
    Detects if the input is a math/logic question and extracts the components.
    Returns: (is_logic, subject, object, relation, goal)
    """
    # Math: "what is X + Y" or "what is X - Y"
    math_match = re.match(r"(?:what is|calculate|compute)?\s*(\d+)\s*([\+\-\*/])\s*(\d+)", user_input, re.IGNORECASE)
    if math_match:
        sub = math_match.group(1)
        goal = math_match.group(2)
        obj = math_match.group(3)
        return True, sub, obj, None, goal
        
    # Analogy: "X is to Y as Z is to..." (Simplified: "dog has bird")
    analogy_match = re.match(r"analogy:\s*(\w+)\s+(\w+)\s+(\w+)", user_input, re.IGNORECASE)
    if analogy_match:
        sub = analogy_match.group(1)
        rel = analogy_match.group(2)
        ctx = analogy_match.group(3)
        return True, sub, None, rel, "analogy", ctx
        
    # Memory Recall: "what did I say about X", "recall X", "what is my favorite X"
    recall_match = re.match(r"(?:what did i say about|recall|what is my favorite)\s+(.*)", user_input, re.IGNORECASE)
    if recall_match:
        query = recall_match.group(1).strip()
        return True, "recall", query, None, None, None
        
    # Memory Store: "remember that X" or "my favorite color is X"
    store_match = re.match(r"(?:remember that|my favorite .* is)\s+(.*)", user_input, re.IGNORECASE)
    if store_match:
        return True, "store", user_input, None, None, None

    return False, None, None, None, None, None

def chat():
    print("=====================================================")
    print("  NEURO-SYMBOLIC HYBRID BRAIN v3.0 - ONLINE")
    print("=====================================================")
    print("Loading Core Topology and Checkpoints...")
    b = brain2.Brain(som_rows=512, som_cols=512, n_dims=512, hidden_dim=512)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
    try:
        b.load_components(
            predictor_path=os.path.join(ckpt_dir, "predictor.bin"),
            language_path=os.path.join(ckpt_dir, "language.bin"),
            som_path=os.path.join(ckpt_dir, "som.bin"),
            episodic_path=os.path.join(ckpt_dir, "episodic.bin"),
            emotion_path=os.path.join(ckpt_dir, "emotion.bin"),
            self_path=os.path.join(ckpt_dir, "self.bin"),
            symbolic_path=os.path.join(ckpt_dir, "symbolic.bin"),
            binding_path=os.path.join(ckpt_dir, "binding.bin"),
            bg_path=os.path.join(ckpt_dir, "bg.bin"),
            procedures_path=os.path.join(ckpt_dir, "procedures.bin"),
            hpred_path=os.path.join(ckpt_dir, "hpred.bin")
        )
        print(f"Successfully loaded 11 cortical hemispheres from {ckpt_dir}!")
    except Exception as e:
        print(f"Failed to load checkpoints: {e}")
        return

    # Load Episodic Text Dictionary
    ep_text_path = os.path.join(ckpt_dir, "episodic_text.json")
    episodic_dict = {}
    if os.path.exists(ep_text_path):
        try:
            with open(ep_text_path, "r") as f:
                episodic_dict = json.load(f)
        except Exception as e:
            print(f"Failed to load episodic text: {e}")

    print("Brain is ready to converse. Type 'quit' to exit.")
    print("Tip: Try asking 'what is 5 + 5' or say 'remember that my dog is brown'.")
    print("Commands: Type '/speak on' or '/speak off' to toggle voice out loud.")
    print("-----------------------------------------------------")
    
    speak_mode = False
    
    while True:
        try:
            user_input = input("\nUser > ").strip().lower()
            if not user_input:
                continue
            if user_input in ['quit', 'exit']:
                break
                
            if user_input == '/speak on':
                speak_mode = True
                print("[System] Voice mode ENABLED.")
                continue
            elif user_input == '/speak off':
                speak_mode = False
                print("[System] Voice mode DISABLED.")
                continue
                
            b.reset_sequence()
            
            # Step 1: Prefrontal Cortex NLP Parsing
            parsed = parse_nlp(user_input)
            is_logic = parsed[0]
            
            if is_logic:
                # --- EXECUTIVE LOGIC ROUTING ---
                action_type = parsed[1]
                
                if action_type == "store":
                    # Manually store explicit memories
                    print(f"[NLP Router] Storing Episodic Memory: '{user_input}'")
                    
                    # Store in C++ Graph
                    start_id = b.episodic.episode_count
                    b.perceive_text(user_input)
                    b.episodic.commit(1.0) # Force commit
                    end_id = b.episodic.episode_count
                    
                    # Store exact text mapping in Python for the entire episode trajectory
                    for i in range(start_id, end_id + 2):
                        episodic_dict[str(i)] = user_input
                    
                    reply = "Got it. I will remember that."
                    print(f"Brain> {reply}")
                    if speak_mode:
                        os.system(f'say "{reply}" &')
                    
                elif action_type == "recall":
                    query = parsed[2]
                    print(f"[NLP Router] Retrieving Memory regarding: '{query}'")
                    
                    for w in query.split():
                        if not b.language.knows(w): b.language.register_word(w)
                    
                    # Push the query into the context map
                    b.scratchpad.write("context_map", b.language.encode(query), "ctx")
                    
                    # Force Op::RETRIEVE (Op index 6)
                    chosen_op = b.force_reason_step(6, "retrieve")
                    
                    # Retrieve the actual text using the top C++ graph index
                    ctx_vec = b.language.encode(query)
                    top_eps = b.episodic.retrieve_topk(ctx_vec, 1)
                    
                    if len(top_eps) > 0:
                        ep_idx = str(top_eps[0][1])
                        if ep_idx in episodic_dict:
                            reply = f"You said: '{episodic_dict[ep_idx]}'"
                        else:
                            # Fallback if text missing but graph node exists
                            res_vec = b.scratchpad.read("result")
                            if len(res_vec) > 0:
                                reply = b.language.best_word(res_vec)
                            else:
                                reply = "I remember something, but the text is fuzzy."
                    else:
                        reply = "I don't have any strong memories of that."
                        
                    print(f"Brain> {reply}")
                    if speak_mode:
                        os.system(f'say "{reply}" &')
                        
                elif len(parsed) == 5: # Math
                    _, sub, obj, _, goal = parsed
                    print(f"[NLP Router] Detected Math: {sub} {goal} {obj}. Routing to Basal Ganglia...")
                    
                    for w in [sub, obj, goal]:
                        if not b.language.knows(w): b.language.register_word(w)
                        
                    b.scratchpad.write("subject", b.language.encode(sub), "ctx")
                    b.scratchpad.write("object", b.language.encode(obj), "ctx")
                    
                    chosen_op = b.direct_reason_step(goal)
                    print(f"[Logic Engine] Basal Ganglia selected Op Index: {chosen_op}")
                    
                    res_vec = b.scratchpad.read("result")
                    if len(res_vec) > 0:
                        reply = b.language.best_word(res_vec)
                    else:
                        reply = "Error: Logic engine returned empty result."
                    print(f"Brain> = {reply}")
                    if speak_mode:
                        os.system(f'say "The answer is {reply}" &')
                        
                elif len(parsed) == 6: # Analogy
                    _, sub, _, rel, goal, ctx = parsed
                    print(f"[NLP Router] Detected Analogy: {sub} {rel} {ctx}. Routing to Basal Ganglia...")
                    
                    for w in [sub, rel, ctx, goal]:
                        if not b.language.knows(w): b.language.register_word(w)
                        
                    b.scratchpad.write("subject", b.language.encode(sub), "ctx")
                    b.scratchpad.write("relation", b.language.encode(rel), "ctx")
                    b.scratchpad.write("context_map", b.language.encode(ctx), "ctx")
                    
                    chosen_op = b.direct_reason_step(goal)
                    print(f"[Logic Engine] Basal Ganglia selected Op Index: {chosen_op}")
                    
                    res_vec = b.scratchpad.read("result")
                    if len(res_vec) > 0:
                        reply = b.language.best_word(res_vec)
                    else:
                        reply = "Error: Logic engine returned empty result."
                    print(f"Brain> = {reply}")
                    if speak_mode:
                        os.system(f'say "The answer is {reply}" &')
                
            else:
                # --- FUZZY CONVERSATIONAL PREDICTION & CONTINUOUS LEARNING ---
                # Perceive the input continuously into the topological map.
                # (Synaptic weights update automatically inside perceive_text via Sparse Plasticity)
                b.perceive_text(user_input)
                
                # Predict the next sequence of words based on semantic space
                res = b.think(6)
                reply = " ".join([w for w in res.words if w])
                print(f"Brain> {reply}")
                
                # Sanitize reply for bash 'say' command
                if speak_mode:
                    safe_reply = reply.replace('"', '').replace("'", "")
                    os.system(f'say "{safe_reply}" &')
            
        except KeyboardInterrupt:
            break
            
    print("\nSaving Checkpoints and Backups...")
    # Create backup directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(os.path.dirname(__file__), "checkpoints", f"executive_brain_backup_{timestamp}")
    shutil.copytree(ckpt_dir, backup_dir)
    print(f"Created isolated backup of previous state at: {backup_dir}")
    
    # Overwrite the live checkpoints with newly learned topological maps and episodic memories
    b.save_components(ckpt_dir)
    
    # Save the Python text dictionary
    with open(os.path.join(ckpt_dir, "episodic_text.json"), "w") as f:
        json.dump(episodic_dict, f)
        
    print("Live Checkpoints safely overwritten with new Continuous Learning updates!")
    
    print("\nShutting down Brain...")

if __name__ == "__main__":
    chat()
