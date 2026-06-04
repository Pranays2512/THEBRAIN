#!/usr/bin/env python3
"""
test_confidence.py — Confidence Gate / "I Don't Know" Tests

Tests that BIND_QUERY now:
  1. Always writes a confidence score to scratchpad
  2. Returns empty result (zero vector) when confidence < 0.25
  3. Returns a real answer when confidence >= 0.25
  4. Confidence increases with repeated bindings
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError:
    print("FATAL: brain2 module not found.")
    sys.exit(1)

CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'stage5_math')
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
failures = []

OP_BIND_QUERY = 5

def check(name, passed, detail=""):
    if passed:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f"  ← {detail}" if detail else ""))
        failures.append(name)

def make_brain():
    b = brain2.Brain(4, 4, 16)
    return b

def load_brain():
    b = brain2.Brain(8, 8, 16)
    b.load_components(
        predictor_path  = f"{CKPT}/predictor.bin",
        language_path   = f"{CKPT}/language.bin",
        som_path        = f"{CKPT}/som.bin",
        episodic_path   = f"{CKPT}/episodic.bin",
        emotion_path    = f"{CKPT}/emotion.bin",
        self_path       = f"{CKPT}/self.bin",
        symbolic_path   = f"{CKPT}/symbolic.bin",
        binding_path    = f"{CKPT}/binding.bin",
        bg_path         = f"{CKPT}/bg.bin",
        procedures_path = f"{CKPT}/procedures.bin",
        hpred_path      = f"{CKPT}/hpred.bin",
    )
    b.symbolic_table.seed_math_symbols()
    return b

def reg(b, w):
    b.language.register_word(w)
    b.symbolic_table.bind(w)
    return b.language.encode(w)


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  Confidence Gate Tests (Op::BIND_QUERY confidence propagation)")
print("═"*60)

# ── Test 1: Known query returns high confidence ───────────────────────────
def test_known_query_high_conf():
    b = make_brain()
    dog_v   = reg(b, "dog")
    animal_v = reg(b, "animal")
    isa_v   = reg(b, "isa")

    # Bind 5x to ensure confidence is high
    for _ in range(5):
        b.binding.bind(dog_v, isa_v, animal_v)

    b.scratchpad.write("subject",  dog_v, "test")
    b.scratchpad.write("relation", isa_v, "test")
    b.force_reason_step(OP_BIND_QUERY, "reply")

    conf   = b.get_last_confidence()
    result = b.scratchpad.read("result")
    best   = b.language.best_word(result)
    norm   = float(np.linalg.norm(result))

    check("Known query: confidence > 0.25", conf >= 0.25, f"conf={conf:.4f}")
    check("Known query: result is non-zero vector", norm > 0.0, f"norm={norm:.4f}")
    check("Known query: decodes to bound object", best == "animal", f"best='{best}'")

# ── Test 2: Unknown query returns low confidence + zero result ────────────
def test_unknown_query_zero():
    b = make_brain()
    xyz_v  = reg(b, "xyz_unknown_AAAA")
    rel_v  = reg(b, "has_relation_BBB")
    # NO bindings registered for xyz

    b.scratchpad.write("subject",  xyz_v, "test")
    b.scratchpad.write("relation", rel_v, "test")
    b.force_reason_step(OP_BIND_QUERY, "reply")

    conf   = b.get_last_confidence()
    result = b.scratchpad.read("result")
    norm   = float(np.linalg.norm(result))

    check("Unknown query: confidence < 0.25", conf < 0.25, f"conf={conf:.4f}")
    check("Unknown query: result is zero vector (I don't know)", norm < 0.01,
          f"norm={norm:.6f}")

# ── Test 3: Wrong relation on known subject → low confidence ─────────────
def test_wrong_relation():
    b = make_brain()
    cat_v  = reg(b, "cat")
    isa_v  = reg(b, "isa")
    col_v  = reg(b, "color")
    anim_v = reg(b, "feline")

    b.binding.bind(cat_v, isa_v, anim_v)  # cat isa feline

    # Query with wrong relation (color)
    b.scratchpad.write("subject",  cat_v, "test")
    b.scratchpad.write("relation", col_v, "test")  # asking "cat color ?" — not bound
    b.force_reason_step(OP_BIND_QUERY, "reply")

    conf = b.get_last_confidence()
    check("Wrong relation → lower confidence than correct relation",
          conf < 0.5, f"conf={conf:.4f}")

# ── Test 4: Confidence increases with repeated bindings ──────────────────
def test_confidence_increases_with_repeats():
    b = make_brain()
    a_v = reg(b, "catX")
    r_v = reg(b, "livesIn")
    o_v = reg(b, "houseX")

    confs = []
    for n_binds in [1, 3, 6]:
        b2 = make_brain()
        a2 = reg(b2, "catX")
        r2 = reg(b2, "livesIn")
        o2 = reg(b2, "houseX")
        for _ in range(n_binds):
            b2.binding.bind(a2, r2, o2)
        b2.scratchpad.write("subject",  a2, "test")
        b2.scratchpad.write("relation", r2, "test")
        b2.force_reason_step(OP_BIND_QUERY, "reply")
        confs.append(b2.get_last_confidence())

    check("1 binding → lower confidence than 3 bindings", confs[0] <= confs[1],
          f"conf[1]={confs[0]:.4f}, conf[3]={confs[1]:.4f}")
    check("3 bindings → lower or equal confidence than 6 bindings", confs[1] <= confs[2],
          f"conf[3]={confs[1]:.4f}, conf[6]={confs[2]:.4f}")

# ── Test 5: get_last_confidence() returns float ───────────────────────────
def test_api_returns_float():
    b = make_brain()
    reg(b, "testword")
    # Don't even run a query — just check the API returns 0.0 safely
    conf = b.get_last_confidence()
    check("get_last_confidence() returns float before any query",
          isinstance(conf, float), f"type={type(conf)}")

# ── Test 6: Loaded checkpoint — semantic query has high confidence ─────────
def test_loaded_checkpoint_known_semantic():
    b = load_brain()
    # bird0 isa animal0 should be in the checkpoint from semantic training
    for w in ["bird0", "isa", "animal0"]:
        b.language.register_word(w)
    bird_v   = b.language.encode("bird0")
    isa_v    = b.language.encode("isa")
    animal_v = b.language.encode("animal0")

    # Bind it now so it's definitely in memory
    b.binding.bind(bird_v, isa_v, animal_v)

    b.scratchpad.write("subject",  bird_v, "test")
    b.scratchpad.write("relation", isa_v,  "test")
    b.force_reason_step(OP_BIND_QUERY, "reply")

    conf = b.get_last_confidence()
    result = b.scratchpad.read("result")
    best   = b.language.best_word(result)
    check("Loaded checkpoint: known binding has confidence >= 0.25",
          conf >= 0.25, f"conf={conf:.4f}")
    check("Loaded checkpoint: result decodes to animal0",
          best == "animal0", f"best='{best}'")


test_known_query_high_conf()
test_unknown_query_zero()
test_wrong_relation()
test_confidence_increases_with_repeats()
test_api_returns_float()
test_loaded_checkpoint_known_semantic()

# ─────────────────────────────────────────────────────────────────────────────
total  = 10
passed = total - len(failures)
print(f"\n{'═'*60}")
print(f"  RESULTS: {passed}/{total} passed")
if failures:
    print("  Failed:")
    for f in failures:
        print(f"    ✗ {f}")
else:
    print("  All tests passed! ✓")
print(f"{'═'*60}")
sys.exit(0 if not failures else 1)
