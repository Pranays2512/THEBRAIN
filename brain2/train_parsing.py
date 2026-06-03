#!/usr/bin/env python3
"""
train_parsing.py — Teach the BG Controller to route incoming words to memory slots.

Problem type:  ["apple", "isa", "fruit"]
Solution path: [STORE_SUBJ, STORE_REL, STORE_OBJ]

The Brain perceives each word sequentially. After each word is perceived, it sits in the `attention` slot (Global Workspace). We train the BG to execute the correct routing operation (`STORE_SUBJ`, `STORE_REL`, or `STORE_OBJ`) so that at the end of the sequence, the scratchpad matches the semantic structure of the fact.

Training signal:
  +1.0  — BG correctly routed all words into the correct slots.
"""

import os, sys, time, random, json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}", flush=True)
    sys.exit(1)

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
SOM_ROWS       = 8
SOM_COLS       = 8
N_DIMS         = 16
N_EPISODES     = 20000         
SAVE_INTERVAL  = 5000          
PRINT_INTERVAL = 1000          
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

OP_NAMES = ["READ", "WRITE", "MATH_SUB", "MATH_DIV", "COMPARE", "BIND_QUERY", "RETRIEVE", "ANALOGY", "HALT", "STORE_SUBJ", "STORE_REL", "STORE_OBJ"]
OP_STORE_SUBJ = 9
OP_STORE_REL  = 10
OP_STORE_OBJ  = 11

FACTS = [
    # Animals
    ("dog", "isa", "animal"), ("cat", "isa", "animal"), ("horse", "isa", "animal"),
    ("lion", "isa", "animal"), ("tiger", "isa", "animal"), ("elephant", "isa", "animal"),
    ("mouse", "isa", "animal"), ("whale", "isa", "animal"), ("shark", "isa", "animal"),
    ("dolphin", "isa", "animal"), ("eagle", "isa", "animal"), ("penguin", "isa", "animal"),
    ("ostrich", "isa", "animal"), ("snake", "isa", "animal"), ("frog", "isa", "animal"),
    
    # Fruits
    ("apple", "isa", "fruit"), ("banana", "isa", "fruit"), ("orange", "isa", "fruit"),
    ("grape", "isa", "fruit"), ("mango", "isa", "fruit"), ("strawberry", "isa", "fruit"),
    ("cherry", "isa", "fruit"), ("peach", "isa", "fruit"), ("pear", "isa", "fruit"),
    ("watermelon", "isa", "fruit"), ("pineapple", "isa", "fruit"),
    
    # Vegetables
    ("carrot", "isa", "vegetable"), ("broccoli", "isa", "vegetable"), ("spinach", "isa", "vegetable"),
    ("potato", "isa", "vegetable"), ("tomato", "isa", "vegetable"), ("onion", "isa", "vegetable"),
    ("garlic", "isa", "vegetable"), ("cucumber", "isa", "vegetable"), ("pepper", "isa", "vegetable"),
    
    # Vehicles
    ("car", "isa", "vehicle"), ("truck", "isa", "vehicle"), ("bus", "isa", "vehicle"),
    ("motorcycle", "isa", "vehicle"), ("bicycle", "isa", "vehicle"), ("train", "isa", "vehicle"),
    ("airplane", "isa", "vehicle"), ("helicopter", "isa", "vehicle"), ("boat", "isa", "vehicle"),
    ("ship", "isa", "vehicle"), ("submarine", "isa", "vehicle"),
    
    # Furniture
    ("chair", "isa", "furniture"), ("table", "isa", "furniture"), ("desk", "isa", "furniture"),
    ("bed", "isa", "furniture"), ("sofa", "isa", "furniture"), ("cabinet", "isa", "furniture"),
    ("bookshelf", "isa", "furniture"), ("couch", "isa", "furniture"),
    
    # Professions
    ("doctor", "isa", "profession"), ("teacher", "isa", "profession"), ("engineer", "isa", "profession"),
    ("lawyer", "isa", "profession"), ("nurse", "isa", "profession"), ("artist", "isa", "profession"),
    ("writer", "isa", "profession"), ("musician", "isa", "profession"), ("actor", "isa", "profession")
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def random_fact():
    return random.choice(FACTS)

def encode_concept(brain, word):
    if not brain.symbolic_table.knows(word):
        brain.symbolic_table.bind(word)
        brain.language.register_word(word)
    return brain.language.encode(word)

def initialize_embeddings(b):
    b.symbolic_table.seed_math_symbols()
    words = ["x", "goal", "relation", "math", "op_symbol", "object", "comparison", "eval", "subject", "result", "binding", "reply", "parse"]
    for subj, rel, obj in FACTS:
        words.extend([subj, rel, obj])
    for word in set(words):
        b.language.register_word(word)
        b.symbolic_table.bind(word)

def cosine(a, b):
    if len(a) == 0 or len(b) == 0: return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return max(0.0, float(np.dot(a, b) / (na * nb)))

# ── Main Training Loop ─────────────────────────────────────────────────────────
def train():
    print("Initializing Brain for Parsing Training...", flush=True)
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS)
    
    # ── Load Stage 3 (Conversation) if available ───────────────────────────────
    stage3_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage3_conversation")
    if os.path.exists(stage3_dir):
        print(f"Loading Stage 3 components from {stage3_dir}...", flush=True)
        b.load_components(
            predictor_path=os.path.join(stage3_dir, "predictor.bin"),
            language_path=os.path.join(stage3_dir, "language.bin"),
            som_path=os.path.join(stage3_dir, "som.bin"),
            episodic_path=os.path.join(stage3_dir, "episodic.bin"),
            emotion_path=os.path.join(stage3_dir, "emotion.bin"),
            self_path=os.path.join(stage3_dir, "self.bin"),
            symbolic_path=os.path.join(stage3_dir, "symbolic.bin"),
            binding_path=os.path.join(stage3_dir, "binding.bin"),
            bg_path=os.path.join(stage3_dir, "bg.bin"), # Keep the BG!
            procedures_path=os.path.join(stage3_dir, "procedures.bin"),
            hpred_path=os.path.join(stage3_dir, "hpred.bin")
        )
        
    initialize_embeddings(b)
    
    CHECKPOINT_DIR_STAGE4 = os.path.join(os.path.dirname(__file__), "checkpoints", "stage4_parsing")
    os.makedirs(CHECKPOINT_DIR_STAGE4, exist_ok=True)
        
    history = {
        "episode": [], "reward": [], "avg_reward_100": [], "solved": []
    }
    
    rewards_window = []
    start_time = time.time()
    
    # Pre-compute target embeddings for fast reward eval
    target_embs = {}
    for w in set([w for f in FACTS for w in f]):
        target_embs[w] = encode_concept(b, w)
        
    for episode in range(1, N_EPISODES + 1):
        subj, rel, obj = random_fact()
        words = [subj, rel, obj]
        expert_seq = [OP_STORE_SUBJ, OP_STORE_REL, OP_STORE_OBJ]
        
        b.reset_sequence()
        goal_vec = np.array(encode_concept(b, "parse"), dtype=np.float32)
        b.scratchpad.write("goal", goal_vec, "goal")
        
        epsilon = max(0.05, 1.0 - (episode / (N_EPISODES * 0.8)))
        is_forcing_episode = (random.random() < epsilon)
        
        ops = []
        b.start_reasoning()
        
        # Sequentially perceive and route
        for step, word in enumerate(words):
            # 1. Perceive the word (puts it into 'attention' slot via GW)
            vec = encode_concept(b, word)
            b.perceive(vec)
            
            # 2. Route it
            if is_forcing_episode:
                op_idx = b.force_reason_step(expert_seq[step], "parse")
            else:
                op_idx = b.direct_reason_step("parse")
                
            ops.append(op_idx)
            
        # Reward Evaluation
        sim_s = cosine(b.scratchpad.read("subject"), target_embs[subj])
        sim_r = cosine(b.scratchpad.read("relation"), target_embs[rel])
        sim_o = cosine(b.scratchpad.read("object"), target_embs[obj])
        
        total_sim = (sim_s + sim_r + sim_o) / 3.0
        
        if total_sim > 0.95:
            reward = 1.0
        else:
            reward = -1.0 + total_sim
            
        if ops == expert_seq:
            reward = max(reward, 0.5)
            
        b.reinforce_bg(reward)
        
        solved = reward >= 1.0
        rewards_window.append(reward)
        if len(rewards_window) > 100:
            rewards_window.pop(0)
        avg_100 = sum(rewards_window) / len(rewards_window)
        
        history["episode"].append(episode)
        history["reward"].append(round(reward, 3))
        history["avg_reward_100"].append(round(avg_100, 4))
        history["solved"].append(int(solved))
        
        if episode % PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            op_seq = [OP_NAMES[o] for o in list(ops)]
            print(f"Ep {episode:5d} | Seq: {words} | Ops: {op_seq} | Reward: {reward:+.2f} | Avg100: {avg_100:+.4f} | {elapsed:.1f}s", flush=True)
            
        if episode % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR_STAGE4, f"bg_ep{episode:06d}.bin")
            b.bg_controller.save(ckpt_path)
            
    print("\n" + "="*60)
    print(f"Training Complete in {time.time() - start_time:.1f}s")
    print(f"Final Avg Reward (last 100): {avg_100:+.4f}")
    solved_count = sum(history["solved"])
    print(f"Fully Solved Episodes: {solved_count} / {N_EPISODES} ({solved_count/N_EPISODES*100:.1f}%)")
    
    print(f"Saving fully integrated Brain to {CHECKPOINT_DIR_STAGE4}...", flush=True)
    b.save_components(CHECKPOINT_DIR_STAGE4)
    print("Stage 4 Complete!", flush=True)

if __name__ == "__main__":
    train()
