import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

def cosine(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return np.dot(v1, v2) / (n1 * n2)

def encode_scalar(brain, value):
    sym = str(value)
    # Using float logic as in the C++ engine
    sym = sym.rstrip('0').rstrip('.') if '.' in sym else sym
    if not brain.symbolic_table.knows(sym):
        brain.symbolic_table.bind(sym)
    return brain.symbolic_table.lookup(sym)

def random_equation():
    a = random.randint(1, 9)
    x = random.randint(1, 9)
    b = random.randint(0, 9)
    c = a * x + b
    return a, b, c, x

def setup_scratchpad(brain, a, b, c):
    pad = brain.scratchpad
    pad.clear()
    
    pad.write("relation", encode_scalar(brain, a), "input")
    pad.write("object", encode_scalar(brain, b), "input")
    pad.write("subject", encode_scalar(brain, c), "input")
    
    goal_vec = np.array(brain.language.encode("x"), dtype=np.float32)
    pad.write("goal", goal_vec, "goal")
    pad.write("comparison", np.zeros(64, dtype=np.float32), "eval")
    
    # ensure result slot is empty but exists in pad memory
    pad.write("result", np.zeros(64, dtype=np.float32), "input")

def train_algebra_step(brain, epsilon):
    a, b_coeff, c, x_true = random_equation()
    setup_scratchpad(brain, a, b_coeff, c)
    
    brain.start_reasoning()
    target_vec = encode_scalar(brain, x_true)
    reward = -1.0
    solved = False
    ops = []
    
    # 2 is MATH_SUB, 0 is READ, 1 is WRITE, 3 is MATH_DIV, 8 is HALT
    expert_seq = [2, 8] if a == 1 else [2, 0, 1, 3, 8]
    
    is_forcing_episode = (random.random() < epsilon)
    for step in range(6):
        
        if is_forcing_episode and (step < len(expert_seq)):
            op_idx = brain.force_reason_step(expert_seq[step], "x")
        else:
            op_idx = brain.reason_step("x", epsilon)
            
        ops.append(op_idx)
        
        result_vec = brain.scratchpad.read("result")
        sim = cosine(result_vec, target_vec)
        if np.isnan(sim):
            sim = 0.0
        
        if op_idx == 8: # HALT
            if sim > 0.95:
                reward = 1.0
                solved = True
            else:
                reward = -1.0
            break
            
    if not solved and reward == -1.0:
        reward = -1.0 + (sim * 0.5)
        
    if ops == expert_seq[:len(ops)]:
        reward = max(reward, 0.5)
        
    brain.reinforce_bg(reward)
    return reward, solved

def train_unified():
    print("Initializing Unified Brain for Algebra & Language Training...")
    b = brain2.Brain(som_rows=10, som_cols=10, n_dims=64, episodic_max=10000)
    
    # Register math vocabulary and numbers
    b.symbolic_table.seed_math_symbols()
    for i in range(101):
        sym = str(i)
        if not b.symbolic_table.knows(sym):
            b.symbolic_table.bind(sym)
    if not b.language.knows("x"):
        b.language.register_word("x")
        
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "simple_stories.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        lines = ["alice went to the store .", "bob likes apples ."]
        
    N_EPISODES = 50000
    
    print("Starting Unified Training Loop...")
    avg_reward = 0.0
    avg_error = 0.0
    
    for episode in range(1, N_EPISODES + 1):
        epsilon = max(0.05, 1.0 - (episode / (N_EPISODES * 0.8)))
        
        # 1. Algebra Task
        alg_reward, solved = train_algebra_step(b, epsilon)
        if np.isnan(alg_reward):
            alg_reward = -1.0
        avg_reward = avg_reward * 0.99 + alg_reward * 0.01
        
        # 2. Language Task
        sentence = random.choice(lines).replace(".", " .").lower()
        words = sentence.split()
        b.reset_sequence()
        err = 0.0
        for w in words:
            word_vec = b.language.encode(w)
            res = b.perceive(word_vec)
            err += res.prediction_error
            # Removing b.reinforce_bg(0.0) to prevent catastrophic forgetting
        avg_error = avg_error * 0.99 + (err / len(words)) * 0.01
        
        # Periodically Dream to consolidate episodic memory
        if episode % 5000 == 0:
            b.dream(n_dreams=5, steps_per_dream=10)
            print(f"Ep {episode:5d} | Alg Reward: {avg_reward:+.4f} | Lang Error: {avg_error:.4f} | Eps: {b.episodic.episode_count}")
            
    print("Unified Training Complete!")
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "bg_ep50000.bin")
    b.bg_controller.save(checkpoint_path)
    print(f"Saved trained BG weights to {checkpoint_path}")

if __name__ == "__main__":
    train_unified()
