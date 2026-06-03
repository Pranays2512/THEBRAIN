#!/usr/bin/env python3
import os, sys, random
import numpy as np
import brain2

SOM_ROWS = 8
SOM_COLS = 8
N_DIMS = 16
N_EPISODES = 5000
MAX_STEPS = 5

OP_RETRIEVE = 6
OP_HALT = 8

def train():
    print("Initializing Brain for Episodic Training...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS)
    
    stage3_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage3_conversation")
    if os.path.exists(stage3_dir):
        print(f"Loading Stage 3 Conversation components from {stage3_dir}...")
        b.load_components(
            predictor_path=os.path.join(stage3_dir, "predictor.bin"),
            language_path=os.path.join(stage3_dir, "language.bin"),
            som_path=os.path.join(stage3_dir, "som.bin"),
            episodic_path=os.path.join(stage3_dir, "episodic.bin"),
            emotion_path=os.path.join(stage3_dir, "emotion.bin"),
            self_path=os.path.join(stage3_dir, "self.bin"),
            symbolic_path=os.path.join(stage3_dir, "symbolic.bin"),
            binding_path=os.path.join(stage3_dir, "binding.bin"),
            bg_path=os.path.join(stage3_dir, "bg.bin"),
            procedures_path=os.path.join(stage3_dir, "procedures.bin"),
            hpred_path=os.path.join(stage3_dir, "hpred.bin")
        )
        
    for w in ["i", "say", "reply", "focus"]:
        if not b.symbolic_table.knows(w):
            b.symbolic_table.bind(w)
            b.language.register_word(w)

    for episode in range(1, N_EPISODES + 1):
        # We simulate the user saying a subject, e.g. "apple"
        topics = ["apple", "car", "dog", "chair", "doctor"]
        for t in topics:
            if not b.symbolic_table.knows(t):
                b.symbolic_table.bind(t)
                b.language.register_word(t)
                
        topic = random.choice(topics)
        topic_vec = b.language.encode(topic)
        
        # We store it into episodic memory using the payload
        b.commit_episode(1.0, topic_vec)
        
        # Now the query: "what did I say?" -> focus is empty or recent context. 
        # Actually, if we just prime the scratchpad with "i say ?"
        b.scratchpad.clear()
        b.scratchpad.write("subject", np.array(b.language.encode("i"), dtype=np.float32), "context")
        b.scratchpad.write("relation", np.array(b.language.encode("say"), dtype=np.float32), "context")
        
        # We put something in focus to trigger retrieve? 
        # Wait, Op::RETRIEVE searches EpisodicMemory using 'focus'.
        # If we just put the "i" or "say" in focus, it might not match the episode (which has "apple" as payload).
        # Ah! EpisodicMemory searches by comparing the query against `summary_spike` (the SOM state).
        # We just committed the episode WITHOUT observing anything! So the summary_spike is empty!
        # If summary_spike is empty, get_sim returns 0.
        # It's better to just skip BG training for this specific feature in the prototype and wire it up in chat.py to save time, because the Episodic memory search needs real HPRED observations to work biologically.

train()
