#!/usr/bin/env python3
"""
teach_numbers.py — Stage 3: Procedural Mastery & Multi-Step Equation Solving

Curriculum:
  1. Register all numbers 0–999 as symbols
  2. Train BG Controller on randomized ax+b=c problems (multi-step)
  3. Train BG Controller on permute/probability/area/power sub-procedures
  4. Save upgraded checkpoint to checkpoints/stage1_32d

Multi-step algebra procedure:
   subject = c, relation = a, object = b
   → MATH_SUB(c - b → result)
   → copy result → subject
   → copy relation → object (divisor)
   → MATH_DIV(result / a → result)
   → SPEAK
   → HALT
"""

import os, sys, random, time, math
import numpy as np
import brain2

# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = "checkpoints/stage1_32d"
CHECKPOINT_OUT = "checkpoints/stage1_32d"
N_EPOCHS       = 3000          # Algebra curriculum epochs
MAX_STEPS      = 8
REINFORCE      = 1.0

# Op codes
OP_READ          = 0;  OP_WRITE         = 1;  OP_MATH_SUB      = 2
OP_MATH_DIV      = 3;  OP_COMPARE       = 4;  OP_BIND_QUERY    = 5
OP_RETRIEVE      = 6;  OP_ANALOGY       = 7;  OP_HALT          = 8
OP_STORE_SUBJ    = 9;  OP_STORE_REL     = 10; OP_STORE_OBJ     = 11
OP_NOT           = 12; OP_BIND_ISA      = 13; OP_ASK_USER      = 14
OP_SPEAK         = 15; OP_ATTEND        = 16; OP_SPEAK_SUBJ    = 17
OP_SPEAK_REL     = 18; OP_SPEAK_OBJ     = 19; OP_MUL           = 21
OP_PERM_N        = 22; OP_PERM_K        = 23; OP_POWER         = 24
OP_DIV_FLOAT     = 26; OP_PREDICT_WM    = 27

# ─────────────────────────────────────────────────────────────────────────────
print("Initializing Brain...", flush=True)
b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32)

if os.path.exists(CHECKPOINT_IN):
    print(f"Loading checkpoint from {CHECKPOINT_IN}...", flush=True)
    b.load_components(
        predictor_path  = f"{CHECKPOINT_IN}/predictor.bin",
        language_path   = f"{CHECKPOINT_IN}/language.bin",
        som_path        = f"{CHECKPOINT_IN}/som.bin",
        episodic_path   = f"{CHECKPOINT_IN}/episodic.bin",
        emotion_path    = f"{CHECKPOINT_IN}/emotion.bin",
        self_path       = f"{CHECKPOINT_IN}/self.bin",
        symbolic_path   = f"{CHECKPOINT_IN}/symbolic.bin",
        binding_path    = f"{CHECKPOINT_IN}/binding.bin",
        bg_path         = f"{CHECKPOINT_IN}/bg.bin",
        procedures_path = f"{CHECKPOINT_IN}/procedures.bin",
        hpred_path      = f"{CHECKPOINT_IN}/hpred.bin",
    )
else:
    print("No checkpoint found; starting fresh.", flush=True)

# Seed math symbols and all numbers 0–999
b.symbolic_table.seed_math_symbols()
print("Registering numbers 0–999...", flush=True)
for i in range(1000):
    w = str(i)
    if not b.language.knows(w):
        b.language.register_word(w)
    b.symbolic_table.bind(w)

# Register operator words
for word in ["probability", "permute", "area", "power", "remember", "solve"]:
    if not b.language.knows(word):
        b.language.register_word(word)
    b.symbolic_table.bind(word)

# ─────────────────────────────────────────────────────────────────────────────
def ensure_word(w):
    if not b.language.knows(w):
        b.language.register_word(w)
    b.symbolic_table.bind(w)

def encode(w):
    ensure_word(w)
    return b.language.encode(w)

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM 1: Multi-step algebra — ax + b = c
# Procedure: MATH_SUB → MATH_DIV → SPEAK → HALT
print("\n═══ CURRICULUM 1: Multi-step Algebra ═══", flush=True)

ALGEBRA_SEQUENCE = [OP_MATH_SUB, OP_MATH_DIV, OP_SPEAK, OP_HALT]
wins = 0
t0 = time.time()

for ep in range(1, N_EPOCHS + 1):
    a_val = random.randint(1, 10)
    b_val = random.randint(0, 50)
    c_val = random.randint(10, 100)

    c_w = str(c_val)
    a_w = str(a_val)
    b_w = str(b_val)
    for w in [c_w, a_w, b_w]: ensure_word(w)

    b.scratchpad.clear()
    b.clear_spoken_words()
    b.scratchpad.write("subject",  encode(c_w), "context")
    b.scratchpad.write("relation", encode(a_w), "context")
    b.scratchpad.write("object",   encode(b_w), "context")
    b.scratchpad.write("goal",     encode("solve"), "goal")
    b.start_reasoning()

    # Teacher forcing
    for op in ALGEBRA_SEQUENCE:
        b.force_reason_step(op, "solve")
        b.reinforce_bg(REINFORCE)

    # Evaluate policy
    b.scratchpad.clear()
    b.clear_spoken_words()
    b.scratchpad.write("subject",  encode(c_w), "context")
    b.scratchpad.write("relation", encode(a_w), "context")
    b.scratchpad.write("object",   encode(b_w), "context")
    b.scratchpad.write("goal",     encode("solve"), "goal")
    b.start_reasoning()

    ops_taken = []
    for _ in range(MAX_STEPS):
        op = b.reason_step("solve", 0.0)
        ops_taken.append(op)
        if op == OP_HALT:
            break

    if ops_taken == ALGEBRA_SEQUENCE:
        wins += 1

    if ep % 300 == 0:
        elapsed = time.time() - t0
        pct = wins / 300 * 100
        print(f"  Ep {ep:4d}/{N_EPOCHS} | WinRate {pct:.0f}% | {elapsed:.1f}s", flush=True)
        wins = 0
        t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM 2: Permutation — N permute K → N!/(N-K)!
# Procedure: PERM_N → PERM_K → SPEAK → HALT  (op 22,23,15,8)
print("\n═══ CURRICULUM 2: Permutation ═══", flush=True)

PERMUTE_SEQUENCE = [OP_PERM_N, OP_PERM_K, OP_SPEAK, OP_HALT]
wins = 0
t0 = time.time()
N_PERM_EPOCHS = 1000

for ep in range(1, N_PERM_EPOCHS + 1):
    n_val = random.randint(3, 7)
    k_val = random.randint(1, n_val)

    n_w = str(n_val); k_w = str(k_val)
    for w in [n_w, k_w]: ensure_word(w)

    b.scratchpad.clear()
    b.clear_spoken_words()
    b.scratchpad.write("subject", encode(n_w), "context")
    b.scratchpad.write("object",  encode(k_w), "context")
    b.scratchpad.write("goal",    encode("permute"), "goal")
    b.start_reasoning()

    for op in PERMUTE_SEQUENCE:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)

    if ep % 200 == 0:
        elapsed = time.time() - t0
        print(f"  Ep {ep:4d}/{N_PERM_EPOCHS} | {elapsed:.1f}s", flush=True)
        t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM 3: Probability — n/d → DIV_FLOAT → SPEAK → HALT
print("\n═══ CURRICULUM 3: Probability ═══", flush=True)

PROB_SEQUENCE = [OP_DIV_FLOAT, OP_SPEAK, OP_HALT]
wins = 0
t0 = time.time()
N_PROB_EPOCHS = 1000

for ep in range(1, N_PROB_EPOCHS + 1):
    n_val = random.randint(1, 50)
    d_val = random.randint(n_val, 100)

    n_w = str(n_val); d_w = str(d_val)
    for w in [n_w, d_w]: ensure_word(w)

    b.scratchpad.clear()
    b.clear_spoken_words()
    b.scratchpad.write("subject", encode(n_w), "context")
    b.scratchpad.write("object",  encode(d_w), "context")
    b.scratchpad.write("goal",    encode("probability"), "goal")
    b.start_reasoning()

    for op in PROB_SEQUENCE:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)

    if ep % 200 == 0:
        elapsed = time.time() - t0
        print(f"  Ep {ep:4d}/{N_PROB_EPOCHS} | {elapsed:.1f}s", flush=True)
        t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM 4: Area — w * h → MUL → SPEAK → HALT
print("\n═══ CURRICULUM 4: Area ═══", flush=True)

AREA_SEQUENCE = [OP_MUL, OP_SPEAK, OP_HALT]
wins = 0
t0 = time.time()
N_AREA_EPOCHS = 1000

for ep in range(1, N_AREA_EPOCHS + 1):
    w_val = random.randint(1, 100)
    h_val = random.randint(1, 100)

    w_w = str(w_val); h_w = str(h_val)
    for w in [w_w, h_w]: ensure_word(w)

    b.scratchpad.clear()
    b.clear_spoken_words()
    b.scratchpad.write("subject", encode(w_w), "context")
    b.scratchpad.write("object",  encode(h_w), "context")
    b.scratchpad.write("goal",    encode("area"), "goal")
    b.start_reasoning()

    for op in AREA_SEQUENCE:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)

    if ep % 200 == 0:
        elapsed = time.time() - t0
        print(f"  Ep {ep:4d}/{N_AREA_EPOCHS} | {elapsed:.1f}s", flush=True)
        t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM 5: Power — b^p → POWER → SPEAK → HALT
print("\n═══ CURRICULUM 5: Power ═══", flush=True)

POWER_SEQUENCE = [OP_POWER, OP_SPEAK, OP_HALT]
wins = 0
t0 = time.time()
N_POWER_EPOCHS = 1000

for ep in range(1, N_POWER_EPOCHS + 1):
    bv = random.randint(1, 10)
    pv = random.randint(0, 4)

    bv_w = str(bv); pv_w = str(pv)
    for w in [bv_w, pv_w]: ensure_word(w)

    b.scratchpad.clear()
    b.clear_spoken_words()
    b.scratchpad.write("subject", encode(bv_w), "context")
    b.scratchpad.write("object",  encode(pv_w), "context")
    b.scratchpad.write("goal",    encode("power"), "goal")
    b.start_reasoning()

    for op in POWER_SEQUENCE:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)

    if ep % 200 == 0:
        elapsed = time.time() - t0
        print(f"  Ep {ep:4d}/{N_POWER_EPOCHS} | {elapsed:.1f}s", flush=True)
        t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Save
print(f"\nSaving to {CHECKPOINT_OUT}...", flush=True)
os.makedirs(CHECKPOINT_OUT, exist_ok=True)
b.save_components(CHECKPOINT_OUT)
print("teach_numbers.py: All curricula complete! ✓", flush=True)
