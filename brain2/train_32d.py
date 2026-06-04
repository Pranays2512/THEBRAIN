"""
train_32d.py — Full 32-dim Training Pipeline

Trains a fresh Brain(10×10, n_dims=32) through 4 stages:
  Stage 1: Math procedures (algebra, permute, probability, area, power)
  Stage 2: World knowledge (semantic ontology + causal chains)
  Stage 3: Self-model
  Stage 4: Grammar/parsing warm-up

Run: PYTHONPATH=. ../venv/bin/python train_32d.py
"""
import brain2
import os, sys, random, math, time

CKPT = "checkpoints/stage1_32d"
os.makedirs(CKPT, exist_ok=True)

# Op codes
OP_MATH_SUB=2; OP_MATH_DIV=3; OP_BIND_QUERY=5; OP_HALT=8
OP_STORE_SUBJ=9; OP_STORE_REL=10; OP_STORE_OBJ=11
OP_BIND_ISA=13; OP_SPEAK=15; OP_MUL=21
OP_PERM_N=22; OP_PERM_K=23; OP_POWER=24; OP_DIV_FLOAT=26
REINFORCE = 1.0

def section(title):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}", flush=True)

# ─── Init ───────────────────────────────────────────────────────────────────
print("Initializing Brain(10×10, n_dims=32)...", flush=True)
b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32)
b.symbolic_table.seed_math_symbols()

for i in range(1000):
    w = str(i)
    if not b.language.knows(w):
        b.language.register_word(w)
    b.symbolic_table.bind(w)

for word in ["probability","permute","area","power","solve","causes","isa","can","has","eats","orbits"]:
    if not b.language.knows(word):
        b.language.register_word(word)
    b.symbolic_table.bind(word)

def enc(w):
    if not b.language.knows(w):
        b.language.register_word(w)
    b.symbolic_table.bind(w)
    return b.language.encode(w)

# ─── STAGE 1A: Algebra ──────────────────────────────────────────────────────
section("Stage 1A: Algebra  ax + b = c  — 3000 epochs")

ALGE_SEQ = [OP_MATH_SUB, OP_MATH_DIV, OP_SPEAK, OP_HALT]
wins = 0; t0 = time.time()

for ep in range(1, 3001):
    a = random.randint(1, 10)
    bv = random.randint(0, 50)
    x = random.randint(-10, 10)
    c = a * x + bv

    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject",  enc(str(c)),  "ctx")
    b.scratchpad.write("relation", enc(str(a)),  "ctx")
    b.scratchpad.write("object",   enc(str(bv)), "ctx")
    b.scratchpad.write("goal",     enc("solve"),  "goal")
    b.start_reasoning()
    for op in ALGE_SEQ:
        b.force_reason_step(op, "solve")
        b.reinforce_bg(REINFORCE)

    # Greedy eval
    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject",  enc(str(c)),  "ctx")
    b.scratchpad.write("relation", enc(str(a)),  "ctx")
    b.scratchpad.write("object",   enc(str(bv)), "ctx")
    b.scratchpad.write("goal",     enc("solve"),  "goal")
    b.start_reasoning()
    ops = []
    for _ in range(8):
        op = b.reason_step("solve", 0.0)
        ops.append(op); 
        if op == OP_HALT: break
    if ops == ALGE_SEQ: wins += 1
    if ep % 500 == 0:
        print(f"  Ep {ep}/3000 | win-rate {wins/500*100:.0f}% | {time.time()-t0:.1f}s", flush=True)
        wins = 0; t0 = time.time()

# ─── STAGE 1B: Permutation ──────────────────────────────────────────────────
section("Stage 1B: Permutation nPk  — 1000 epochs")

PERM_SEQ = [OP_PERM_N, OP_PERM_K, OP_SPEAK, OP_HALT]
for ep in range(1, 1001):
    n = random.randint(3, 7); k = random.randint(1, n)
    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject", enc(str(n)), "ctx")
    b.scratchpad.write("object",  enc(str(k)), "ctx")
    b.scratchpad.write("goal",    enc("permute"), "goal")
    b.start_reasoning()
    for op in PERM_SEQ:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)
    if ep % 200 == 0:
        print(f"  Ep {ep}/1000 | {time.time()-t0:.1f}s", flush=True); t0 = time.time()

# ─── STAGE 1C: Probability ──────────────────────────────────────────────────
section("Stage 1C: Probability n/d  — 1000 epochs")

PROB_SEQ = [OP_DIV_FLOAT, OP_SPEAK, OP_HALT]
for ep in range(1, 1001):
    d = random.randint(2, 100); n = random.randint(1, d)
    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject", enc(str(n)), "ctx")
    b.scratchpad.write("object",  enc(str(d)), "ctx")
    b.scratchpad.write("goal",    enc("probability"), "goal")
    b.start_reasoning()
    for op in PROB_SEQ:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)
    if ep % 200 == 0:
        print(f"  Ep {ep}/1000 | {time.time()-t0:.1f}s", flush=True); t0 = time.time()

# ─── STAGE 1D: Area ─────────────────────────────────────────────────────────
section("Stage 1D: Area w×h  — 1000 epochs")

AREA_SEQ = [OP_MUL, OP_SPEAK, OP_HALT]
for ep in range(1, 1001):
    w = random.randint(1, 100); h = random.randint(1, 100)
    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject", enc(str(w)), "ctx")
    b.scratchpad.write("object",  enc(str(h)), "ctx")
    b.scratchpad.write("goal",    enc("area"), "goal")
    b.start_reasoning()
    for op in AREA_SEQ:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)
    if ep % 200 == 0:
        print(f"  Ep {ep}/1000 | {time.time()-t0:.1f}s", flush=True); t0 = time.time()

# ─── STAGE 1E: Power ────────────────────────────────────────────────────────
section("Stage 1E: Power b^p  — 1000 epochs")

POWER_SEQ = [OP_POWER, OP_SPEAK, OP_HALT]
for ep in range(1, 1001):
    base = random.randint(1, 10); exp = random.randint(0, 4)
    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject", enc(str(base)), "ctx")
    b.scratchpad.write("object",  enc(str(exp)),  "ctx")
    b.scratchpad.write("goal",    enc("power"), "goal")
    b.start_reasoning()
    for op in POWER_SEQ:
        b.force_reason_step(op, "reply")
        b.reinforce_bg(REINFORCE)
    if ep % 200 == 0:
        print(f"  Ep {ep}/1000 | {time.time()-t0:.1f}s", flush=True); t0 = time.time()

b.save_components(CKPT)
print(f"\n✓ Stage 1 (Math) saved to {CKPT}", flush=True)

# ─── STAGE 2: World Knowledge ───────────────────────────────────────────────
section("Stage 2: World Knowledge — 75 facts + 500 BG epochs")

world_facts = [
    # Animals
    ("dog","isa","animal"), ("cat","isa","animal"), ("bird","isa","animal"),
    ("fish","isa","animal"), ("lion","isa","animal"), ("elephant","isa","animal"),
    ("dog","has","legs"), ("bird","has","wings"), ("fish","has","fins"),
    ("dog","can","bark"), ("bird","can","fly"), ("fish","can","swim"),
    ("lion","eats","meat"), ("dog","eats","food"), ("cat","eats","fish"),
    # Plants
    ("tree","isa","plant"), ("rose","isa","plant"), ("grass","isa","plant"),
    ("flower","isa","plant"), ("tree","has","roots"), ("rose","has","thorns"),
    ("tree","can","grow"), ("flower","has","petals"),
    # People
    ("mother","isa","parent"), ("father","isa","parent"),
    ("parent","isa","human"), ("child","isa","human"), ("human","isa","animal"),
    ("doctor","isa","human"), ("teacher","isa","human"),
    ("teacher","can","teach"), ("doctor","can","heal"),
    # Cosmos
    ("sun","isa","star"), ("earth","isa","planet"),
    ("moon","orbits","earth"), ("earth","orbits","sun"),
    # Elements
    ("fire","has","heat"), ("water","isa","liquid"),
    ("ice","isa","solid"), ("water","can","freeze"),
    # Causal chains
    ("rain","causes","wet"), ("wet","causes","cold"),
    ("cold","causes","illness"), ("heat","causes","sweat"),
    ("study","causes","knowledge"), ("knowledge","causes","wisdom"),
    ("sleep","causes","rest"), ("rest","causes","energy"),
    ("energy","causes","action"), ("action","causes","result"),
    ("fire","causes","heat"), ("ice","causes","cold"),
    # Colors
    ("red","isa","color"), ("blue","isa","color"), ("green","isa","color"),
    ("sky","has","blue"), ("grass","has","green"), ("fire","has","red"),
    # Food
    ("apple","isa","fruit"), ("banana","isa","fruit"),
    ("bread","isa","food"), ("milk","isa","food"), ("fruit","isa","food"),
    ("apple","has","seeds"),
    # Places
    ("school","isa","place"), ("hospital","isa","place"),
    ("home","isa","place"), ("city","isa","place"),
    ("school","has","teacher"), ("hospital","has","doctor"),
    # Self
    ("i","isa","ai"), ("i","can","learn"), ("i","can","remember"),
    ("i","can","reason"), ("brain","isa","ai"), ("brain","can","think"),
]

for subj, rel, obj in world_facts:
    for w in [subj, rel, obj]:
        if not b.language.knows(w): b.language.register_word(w)
        b.symbolic_table.bind(w)
    sv = enc(subj); rv = enc(rel); ov = enc(obj)
    for _ in range(3):
        b.binding.bind(sv, rv, ov)
    b.perceive(sv)
    b.commit_episode(1.0, sv[:32])

print(f"  Bound {len(world_facts)} world facts", flush=True)

# BG training on semantic queries
wins = 0
for ep in range(1, 501):
    subj, rel, obj = random.choice(world_facts)
    sv = enc(subj); rv = enc(rel)
    b.scratchpad.clear(); b.clear_spoken_words()
    b.scratchpad.write("subject",  sv, "sem")
    b.scratchpad.write("relation", rv, "sem")
    b.scratchpad.write("goal",     enc("reply"), "goal")
    b.start_reasoning()
    b.force_reason_step(OP_BIND_QUERY, "reply")
    b.force_reason_step(OP_SPEAK,      "reply")
    b.force_reason_step(OP_HALT,       "reply")
    spoken = b.get_spoken_words(); b.clear_spoken_words()
    reward = 1.0 if spoken and spoken[-1] == obj else -0.3
    if reward > 0: wins += 1
    b.reinforce_bg(reward)
    if ep % 100 == 0:
        print(f"  Ep {ep}/500 | semantic win {wins/100*100:.0f}%", flush=True)
        wins = 0

b.save_components(CKPT)
print(f"\n✓ Stage 2 (World) saved to {CKPT}", flush=True)

# ─── STAGE 3: Grammar warm-up ───────────────────────────────────────────────
section("Stage 3: Grammar / Parsing  — 500 sentences")

sentences = [
    ("dog","isa","animal"), ("cat","isa","animal"), ("bird","can","fly"),
    ("fish","can","swim"), ("teacher","can","teach"), ("doctor","can","heal"),
    ("sun","isa","star"), ("moon","orbits","earth"), ("apple","isa","fruit"),
    ("water","isa","liquid"), ("fire","has","heat"), ("rain","causes","wet"),
]

for ep in range(1, 501):
    subj, rel, obj = random.choice(sentences)
    for w, slot, op in [(subj,"subject",OP_STORE_SUBJ),(rel,"relation",OP_STORE_REL),(obj,"object",OP_STORE_OBJ)]:
        vec = enc(w)
        b.perceive(vec)
        b.scratchpad.write(slot, vec, "parse")
        b.start_reasoning()
        b.force_reason_step(op, "parse")
    b.force_reason_step(OP_BIND_ISA, "store")
    b.reinforce_bg(1.0)
    if ep % 100 == 0:
        print(f"  Ep {ep}/500", flush=True)

b.save_components(CKPT)
print(f"\n✓ Stage 3 (Grammar) saved to {CKPT}", flush=True)

# ─── DONE ───────────────────────────────────────────────────────────────────
section("Training Complete!")
print(f"  Brain:      10×10 SOM,  n_dims=32  ({10*10} neurons, 2× representational capacity)")
print(f"  Checkpoint: {CKPT}/")
print(f"  Stages:     Math (5 curricula) → World ({len(world_facts)} facts) → Grammar")
print(f"\n  Next:")
print(f"    PYTHONPATH=. ../venv/bin/python tests/run_hardened_suite.py")
print(f"    PYTHONPATH=. ../venv/bin/python server.py")
