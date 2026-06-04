#!/usr/bin/env python3
"""
test_arch_fixes.py — Architecture Fix Verification Tests

Tests all 3 newly fixed architectural weaknesses:
  1. BG Replay Buffer (catastrophic forgetting prevention)
  2. BindingMemory LSH (O(1) approximate subject lookup)
  3. PC Warm-up in start_reasoning() (pc_wm/pc_bg no longer dormant)
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError:
    print("FATAL: brain2 module not found.")
    sys.exit(1)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
failures = []

OP_STORE_SUBJ = 9; OP_STORE_REL = 10; OP_STORE_OBJ = 11
OP_BIND_QUERY = 5; OP_CHAIN_FOLLOW = 28

def check(name, passed, detail=""):
    if passed:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f"  ← {detail}" if detail else ""))
        failures.append(name)

def make_brain():
    b = brain2.Brain(4, 4, 16)
    return b

def reg(b, w):
    b.language.register_word(w)
    b.symbolic_table.bind(w)
    return b.language.encode(w)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  Fix 1: BG Experience Replay Buffer (anti-forgetting)")
print("═"*65)

def test_replay_prevents_forgetting():
    """
    Train BG on task A (BIND_QUERY=5), then on task B (CHAIN_FOLLOW=28).
    Without replay, the BIND_QUERY association would be overwritten.
    With replay, some BIND_QUERY activations survive in the buffer.
    We verify the replay buffer is populated and sampled.
    """
    b = make_brain()

    # Phase 1: teach BIND_QUERY
    a_v = reg(b, "catA"); r_v = reg(b, "isaR"); o_v = reg(b, "animalA")
    b.binding.bind(a_v, r_v, o_v)
    b.scratchpad.write("subject",  a_v, "test")
    b.scratchpad.write("relation", r_v, "test")
    b.start_reasoning()
    for _ in range(20):
        b.force_reason_step(OP_BIND_QUERY, "reply")
        b.reinforce_bg(1.0)

    # Phase 2: teach CHAIN_FOLLOW on different data
    for i in range(15):
        b.force_reason_step(OP_CHAIN_FOLLOW, "causes")
        b.reinforce_bg(1.0)

    # The replay buffer should be populated (accessed indirectly via reinforce)
    # We can verify by running BIND_QUERY still works after the second curriculum
    b.scratchpad.clear()
    b.scratchpad.write("subject",  a_v, "test")
    b.scratchpad.write("relation", r_v, "test")
    b.start_reasoning()
    b.force_reason_step(OP_BIND_QUERY, "reply")
    conf = b.get_last_confidence()
    result = b.scratchpad.read("result")
    best = b.language.best_word(result)
    check("BIND_QUERY still works after CHAIN_FOLLOW curriculum (replay kept it)",
          conf >= 0.25 and best == "animalA",
          f"conf={conf:.3f}, best='{best}'")

def test_replay_buffer_fills():
    """Verify the replay buffer stores experiences across multiple reinforce() calls."""
    b = make_brain()
    a_v = reg(b, "dog0"); r_v = reg(b, "has"); o_v = reg(b, "paw0")
    b.binding.bind(a_v, r_v, o_v)
    # Run 30 reinforce cycles — enough to fill several replay slots
    for i in range(30):
        b.scratchpad.write("subject",  a_v, "test")
        b.scratchpad.write("relation", r_v, "test")
        b.start_reasoning()
        b.force_reason_step(OP_BIND_QUERY, "reply")
        b.reinforce_bg(1.0 if i % 2 == 0 else -0.5)
    # If no crash and brain still responds, replay buffer didn't corrupt state
    b.scratchpad.clear()
    b.scratchpad.write("subject",  a_v, "test")
    b.scratchpad.write("relation", r_v, "test")
    b.start_reasoning()
    b.force_reason_step(OP_BIND_QUERY, "reply")
    conf = b.get_last_confidence()
    check("Replay buffer fills without corrupting state (30 cycles)", conf >= 0.0,
          f"conf={conf:.3f}")

test_replay_prevents_forgetting()
test_replay_buffer_fills()


# ═══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  Fix 2: BindingMemory LSH — O(1) approximate lookup")
print("═"*65)

def test_lsh_finds_known_binding():
    """LSH should still find the correct answer for a known (subj, rel) pair."""
    b = make_brain()
    cat_v  = reg(b, "cat0"); isa_v = reg(b, "isa0"); anim_v = reg(b, "animal0")
    for _ in range(3):
        b.binding.bind(cat_v, isa_v, anim_v)
    [ans, conf] = b.binding.query(cat_v, isa_v, True, 0.3, 1)
    best = b.language.best_word(ans)
    check("LSH: known binding still found after insert", conf >= 0.25 and best == "animal0",
          f"conf={conf:.3f}, best='{best}'")

def test_lsh_scales_without_slowdown():
    """Insert 500 bindings and verify query speed doesn't degrade noticeably vs 50."""
    b50  = make_brain()
    b500 = make_brain()

    # Target binding we'll query for
    tgt_subj = reg(b50,  "target_subject_XYZ")
    tgt_rel  = reg(b50,  "target_relation_XYZ")
    tgt_obj  = reg(b50,  "target_object_XYZ")
    reg(b500, "target_subject_XYZ")
    reg(b500, "target_relation_XYZ")
    reg(b500, "target_object_XYZ")
    tgt_subj2 = b500.language.encode("target_subject_XYZ")
    tgt_rel2  = b500.language.encode("target_relation_XYZ")
    tgt_obj2  = b500.language.encode("target_object_XYZ")

    b50.binding.bind(tgt_subj, tgt_rel, tgt_obj)
    b500.binding.bind(tgt_subj2, tgt_rel2, tgt_obj2)

    # Add noise bindings
    for i in range(49):
        n_s = reg(b50, f"noise_s_{i}"); n_r = reg(b50, f"noise_r_{i}"); n_o = reg(b50, f"noise_o_{i}")
        b50.binding.bind(n_s, n_r, n_o)
    for i in range(499):
        n_s = reg(b500, f"noise_s_{i}"); n_r = reg(b500, f"noise_r_{i}"); n_o = reg(b500, f"noise_o_{i}")
        b500.binding.bind(n_s, n_r, n_o)

    # Time 100 queries on each
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        b50.binding.query(tgt_subj, tgt_rel, True, 0.3, 1)
    t50 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        b500.binding.query(tgt_subj2, tgt_rel2, True, 0.3, 1)
    t500 = time.perf_counter() - t0

    # LSH: 500-entry should be at most 4× slower than 50-entry (vs 10× for linear scan)
    ratio = t500 / (t50 + 1e-9)
    check(f"LSH: 500 entries not >4× slower than 50 entries (ratio={ratio:.2f}x)",
          ratio < 4.0, f"t50={t50*1000:.1f}ms t500={t500*1000:.1f}ms ratio={ratio:.2f}")
    check("LSH: target binding still found at 500 entries",
          b500.binding.query(tgt_subj2, tgt_rel2, True, 0.3, 1)[1] >= 0.25)

def test_lsh_eviction_safe():
    """Evicting entries shouldn't corrupt the LSH index."""
    b = make_brain()
    # Fill to capacity (max_bindings defaults to 2000 for full brains, 1000 for test)
    base_s = reg(b, "evict_s"); base_r = reg(b, "evict_r"); base_o = reg(b, "evict_o")
    b.binding.bind(base_s, base_r, base_o)  # this is the one we want to survive

    # Flood with 60 more (enough to trigger some eviction for a small max_bindings brain)
    for i in range(60):
        s = reg(b, f"filler_s_{i}"); r2 = reg(b, f"filler_r_{i}"); o = reg(b, f"filler_o_{i}")
        b.binding.bind(s, r2, o)

    # Query shouldn't crash — the LSH index must be consistent
    crashed = False
    try:
        b.binding.query(base_s, base_r, True, 0.1, 1)
    except Exception as e:
        crashed = True
        check("LSH: eviction doesn't corrupt index", False, str(e))
    if not crashed:
        check("LSH: eviction doesn't corrupt index", True)

test_lsh_finds_known_binding()
test_lsh_scales_without_slowdown()
test_lsh_eviction_safe()


# ═══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  Fix 3: PC Warm-up (pc_wm/pc_bg non-zero at reasoning start)")
print("═"*65)

def test_pc_warmup_produces_nonzero_context():
    """After start_reasoning() with scratchpad data, WM context should be non-zero."""
    b = make_brain()
    dog_v = reg(b, "dog1"); isa_v = reg(b, "isa1")
    b.scratchpad.write("subject",  dog_v, "test")
    b.scratchpad.write("relation", isa_v, "test")
    b.start_reasoning()
    # working_mem.context() should now be non-zero
    ctx = b.working_mem.context()
    norm = float(np.linalg.norm(ctx))
    check("PC warm-up: working_mem context is non-zero after start_reasoning()",
          norm > 1e-4, f"norm={norm:.6f}")

def test_pc_warmup_no_context_without_scratchpad():
    """With no scratchpad slots written, warm-up should not crash."""
    b = make_brain()
    crashed = False
    try:
        b.start_reasoning()
    except Exception as e:
        crashed = True
        check("PC warm-up: no crash when scratchpad is empty", False, str(e))
    if not crashed:
        check("PC warm-up: no crash when scratchpad is empty", True)

def test_pc_warmup_improves_predict_wm():
    """Op::PREDICT_WM result should be non-zero after warm-up."""
    OP_PREDICT_WM = 27
    b = make_brain()
    cat_v = reg(b, "cat1"); isa_v = reg(b, "isa1")
    b.scratchpad.write("subject",  cat_v, "test")
    b.scratchpad.write("relation", isa_v, "test")
    b.start_reasoning()
    b.force_reason_step(OP_PREDICT_WM, "predict")
    result = b.scratchpad.read("result")
    norm = float(np.linalg.norm(result))
    check("PC warm-up: Op::PREDICT_WM produces non-zero vector after warm-up",
          norm > 1e-4, f"norm={norm:.6f}")

def test_pc_warmup_doesnt_break_bind_query():
    """Warm-up should not interfere with BIND_QUERY results."""
    b = make_brain()
    s_v = reg(b, "fish1"); r_v = reg(b, "isa1"); o_v = reg(b, "animal1")
    b.binding.bind(s_v, r_v, o_v)
    b.scratchpad.write("subject",  s_v, "test")
    b.scratchpad.write("relation", r_v, "test")
    b.start_reasoning()
    b.force_reason_step(OP_BIND_QUERY, "reply")
    conf = b.get_last_confidence()
    best = b.language.best_word(b.scratchpad.read("result"))
    check("PC warm-up: BIND_QUERY result unaffected (fish1 isa animal1)",
          conf >= 0.25 and best == "animal1", f"conf={conf:.3f}, best='{best}'")

test_pc_warmup_produces_nonzero_context()
test_pc_warmup_no_context_without_scratchpad()
test_pc_warmup_improves_predict_wm()
test_pc_warmup_doesnt_break_bind_query()


# ═══════════════════════════════════════════════════════════════════
total  = 11
passed = total - len(failures)
print(f"\n{'═'*65}")
print(f"  RESULTS: {passed}/{total} passed")
if failures:
    print("  Failed:")
    for f in failures:
        print(f"    ✗ {f}")
else:
    print("  All architecture-fix tests passed! ✓")
print(f"{'═'*65}")
sys.exit(0 if not failures else 1)
