#!/usr/bin/env python3
"""
train_conversation.py — Teach the BG Controller to perform memory retrieval.

Problem type:  "X isa ?"  →  retrieve Y
Solution path: BIND_QUERY -> HALT

We set up the scratchpad with symbolic slots representing the query,
then call brain.reason("Y") and reward/penalise based on whether the
returned operation sequence is coherent with the correct solution steps.

Training signal (TD-lambda via reinforce_bg):
  +1.0  — BG chose BIND_QUERY then HALT: correct!
  -1.0  — BG chose random/wrong sequence
   0.0  — BG reached HALT without solving
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
N_EPISODES     = 50000         # Training episodes
MAX_STEPS      = 6             # Max reasoning steps per episode
SAVE_INTERVAL  = 5000          # Save BG weights every N episodes
PRINT_INTERVAL = 1000          # Print progress every N episodes
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

OP_NAMES = ["READ", "WRITE", "MATH_SUB", "MATH_DIV", "COMPARE", "BIND_QUERY", "RETRIEVE", "ANALOGY", "HALT", "STORE_SUBJ", "STORE_REL", "STORE_OBJ", "NOT", "BIND_ISA"]
OP_READ = 0
OP_WRITE = 1
OP_COMPARE = 4
OP_BIND_QUERY = 5
OP_HALT  = 8
OP_BIND_ISA = 13

# Fact base
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
    ("writer", "isa", "profession"), ("musician", "isa", "profession"), ("actor", "isa", "profession"),
    
    # Has Relationships
    ("animal", "has", "dna"),
    ("fruit", "has", "seeds"),
    ("vegetable", "has", "vitamins"),
    ("vehicle", "has", "wheels"),
    ("furniture", "has", "legs"),
    ("profession", "has", "salary")
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def random_query():
    """Pick a random fact."""
    return random.choice(FACTS)

def encode_concept(brain, word):
    """Encode a string concept into a language vector."""
    if not brain.symbolic_table.knows(word):
        brain.symbolic_table.bind(word)
        brain.language.register_word(word)
    return brain.language.encode(word)

def initialize_embeddings(b):
    b.symbolic_table.seed_math_symbols()
    words = ["x", "goal", "relation", "math", "op_symbol", "object", "comparison", "eval", "subject", "result", "binding", "reply"]
    
    # Add words from facts
    for subj, rel, obj in FACTS:
        words.extend([subj, rel, obj])
        
    for word in set(words):
        b.language.register_word(word)
        b.symbolic_table.bind(word)
        
    for subj, rel, obj in FACTS:
        s_vec = b.language.encode(subj)
        r_vec = b.language.encode(rel)
        o_vec = b.language.encode(obj)
        b.bind_triple(s_vec, r_vec, o_vec)

def setup_scratchpad(brain, subj, rel):
    """
    Prime the scratchpad with the query slots.
    
    subject  = vec("apple")
    relation = vec("isa")
    goal     = vec("reply")
    """
    brain.scratchpad.clear()
    brain.scratchpad.write("subject",  np.array(encode_concept(brain, subj),  dtype=np.float32), "context")
    brain.scratchpad.write("relation", np.array(encode_concept(brain, rel),  dtype=np.float32), "context")
    
    goal_vec = np.array(encode_concept(brain, "reply"), dtype=np.float32)
    brain.scratchpad.write("goal",     goal_vec, "goal")
    brain.scratchpad.write("comparison", np.zeros(N_DIMS, dtype=np.float32), "eval")

def cosine(a, b):
    if len(a) == 0 or len(b) == 0: return 0.0
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return max(0.0, float(np.dot(a, b) / (na * nb)))

# ── Main Training Loop ─────────────────────────────────────────────────────────
def train():
    print("Initializing Brain for Conversation Training...", flush=True)
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS)
    
    # ── Load Stage 2 (Math) if available ───────────────────────────────
    stage2_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage2_math")
    if os.path.exists(stage2_dir):
        print(f"Loading Stage 2 Math components from {stage2_dir}...", flush=True)
        b.load_components(
            predictor_path=os.path.join(stage2_dir, "predictor.bin"),
            language_path=os.path.join(stage2_dir, "language.bin"),
            som_path=os.path.join(stage2_dir, "som.bin"),
            episodic_path=os.path.join(stage2_dir, "episodic.bin"),
            emotion_path=os.path.join(stage2_dir, "emotion.bin"),
            self_path=os.path.join(stage2_dir, "self.bin"),
            symbolic_path=os.path.join(stage2_dir, "symbolic.bin"),
            binding_path=os.path.join(stage2_dir, "binding.bin"),
            bg_path=os.path.join(stage2_dir, "bg.bin"), # Load BG to keep math skills!
            procedures_path=os.path.join(stage2_dir, "procedures.bin"),
            hpred_path=os.path.join(stage2_dir, "hpred.bin")
        )
        
    initialize_embeddings(b)
    
    CHECKPOINT_DIR_STAGE3 = os.path.join(os.path.dirname(__file__), "checkpoints", "stage3_conversation")
    os.makedirs(CHECKPOINT_DIR_STAGE3, exist_ok=True)
    
    # Pre-load facts into Binding Memory
    for subj, rel, obj in FACTS:
        subj_vec = encode_concept(b, subj)
        rel_vec = encode_concept(b, rel)
        obj_vec = encode_concept(b, obj)
        b.bind_triple(subj_vec, rel_vec, obj_vec)
        
    history = {
        "episode": [], "reward": [], "avg_reward_100": [],
        "solved": []
    }
    
    rewards_window = []
    start_time = time.time()
    
    for episode in range(1, N_EPISODES + 1):
        # 33% chance of Retrieval ("X isa ?")
        # 33% chance of Verification ("X isa Y ?")
        # 33% chance of Multihop ("X has ?")
        rand_val = random.random()
        if rand_val < 0.33:
            task_type = "retrieval"
        elif rand_val < 0.66:
            task_type = "verification"
        else:
            task_type = "multihop"
            
        if task_type == "multihop":
            # For multihop, we need to pick a subject that has an 'isa' relation to something that has a 'has' relation.
            # E.g. "dog" isa "animal", "animal" has "dna" -> "dog" has "dna"
            # We find a valid chain:
            valid_chains = []
            for s1, r1, o1 in FACTS:
                if r1 == "isa":
                    for s2, r2, o2 in FACTS:
                        if s2 == o1 and r2 == "has":
                            valid_chains.append((s1, o2)) # (dog, dna)
            
            if not valid_chains:
                task_type = "retrieval" # Fallback if no chains (shouldn't happen)
            else:
                subj, target_obj = random.choice(valid_chains)
                rel = "has"
                
                b.scratchpad.clear()
                b.scratchpad.write("subject",  np.array(encode_concept(b, subj),  dtype=np.float32), "context")
                b.scratchpad.write("relation", np.array(encode_concept(b, rel),  dtype=np.float32), "context")
                
                goal_vec = np.array(encode_concept(b, "reply"), dtype=np.float32)
                b.scratchpad.write("goal",     goal_vec, "goal")
                
                # expert_seq: BIND_QUERY (fails), BIND_ISA(dog) -> animal, READ(animal), WRITE(animal->subj), BIND_QUERY(animal, has)->dna, HALT
                expert_seq = [OP_BIND_QUERY, OP_BIND_ISA, OP_READ, OP_WRITE, OP_BIND_QUERY, OP_HALT]
                goal_word = "reply"
                obj = target_obj # For reward calculation
                
        if task_type == "verification":
            subj, rel, obj = random_query()
            # 50% chance true, 50% chance false
            is_true = random.random() < 0.5
            verify_obj = obj if is_true else random_query()[2]
            
            b.scratchpad.clear()
            b.scratchpad.write("subject",  np.array(encode_concept(b, subj),  dtype=np.float32), "context")
            b.scratchpad.write("relation", np.array(encode_concept(b, rel),  dtype=np.float32), "context")
            b.scratchpad.write("object",   np.array(encode_concept(b, verify_obj), dtype=np.float32), "context")
            
            goal_vec = np.array(encode_concept(b, "eval"), dtype=np.float32)
            b.scratchpad.write("goal",     goal_vec, "goal")
            b.scratchpad.write("comparison", np.zeros(N_DIMS, dtype=np.float32), "eval")
            
            expert_seq = [OP_BIND_QUERY, OP_COMPARE, OP_HALT]
            goal_word = "eval"
        elif task_type == "retrieval":
            subj, rel, obj = random_query()
            setup_scratchpad(b, subj, rel)
            expert_seq = [OP_BIND_QUERY, OP_HALT]
            goal_word = "reply"
        
        epsilon = max(0.05, 1.0 - (episode / (N_EPISODES * 0.8)))

        b.start_reasoning()
        ops = []
        reward = -1.0
        solved = False
        
        is_forcing_episode = (random.random() < epsilon)
        for step in range(MAX_STEPS):
            if is_forcing_episode and (step < len(expert_seq)):
                op_idx = b.force_reason_step(expert_seq[step], goal_word)
            else:
                op_idx = b.direct_reason_step(goal_word)
            
            ops.append(op_idx)
            
            if task_type == "retrieval" or task_type == "multihop":
                target_vec = np.array(encode_concept(b, obj), dtype=np.float32)
                result_vec = b.scratchpad.read("result")
                sim = cosine(result_vec, target_vec)
                if sim > 0.95:
                    reward = 1.0
                    solved = True
                    break
            else:
                # For verification, just check if it executed BIND_QUERY then COMPARE
                if ops == expert_seq:
                    reward = 1.0
                    solved = True
                    break
                sim = 0.0 # no partial sim reward for verification
                
            if op_idx == OP_HALT:
                reward = -1.0
                break
                
        if not solved and reward == -1.0 and task_type == "retrieval":
            reward = -1.0 + (sim * 0.5)
            
        if ops == expert_seq[:len(ops)]:
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
            print(f"Ep {episode:5d} | Query: {subj} {rel} ? (A: {obj}) | "
                  f"Ops: {op_seq} | Reward: {reward:+.2f} | "
                  f"Avg100: {avg_100:+.4f} | {elapsed:.1f}s", flush=True)
        
        if episode % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR_STAGE3, f"bg_ep{episode:06d}.bin")
            b.bg_controller.save(ckpt_path)
    
    print("\n" + "="*60)
    print(f"Training Complete in {time.time() - start_time:.1f}s")
    print(f"Episodes: {N_EPISODES}")
    print(f"Final Avg Reward (last 100): {avg_100:+.4f}")
    solved_count = sum(history["solved"])
    print(f"Fully Solved Episodes: {solved_count} / {N_EPISODES} ({solved_count/N_EPISODES*100:.1f}%)")
    
    print(f"Saving fully integrated Brain to {CHECKPOINT_DIR_STAGE3}...", flush=True)
    b.save_components(CHECKPOINT_DIR_STAGE3)
    print("Stage 3 Complete!", flush=True)
    
    return b, history

if __name__ == "__main__":
    train()
