#!/usr/bin/env python3
"""
test_multihop.py — Multi-Hop Causal Chain Reasoning Tests

Tests Op::CHAIN_FOLLOW (op=28) for iterative BFS traversal
along a relation through BindingMemory, supporting chains up to depth 10.
"""
import os, sys
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

OP_CHAIN_FOLLOW = 28
OP_BIND_QUERY   = 5

def check(name, passed, detail=""):
    if passed:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f"  ← {detail}" if detail else ""))
        failures.append(name)


def make_brain():
    b = brain2.Brain(4, 4, 16)
    return b

def register(b, *words):
    for w in words:
        b.language.register_word(w)
        b.symbolic_table.bind(w)

def enc(b, w):
    register(b, w)
    return b.language.encode(w)

def bind_chain(b, nodes, rel_word):
    """Bind node[i] --rel--> node[i+1] for all consecutive pairs."""
    rel_v = enc(b, rel_word)
    for i in range(len(nodes) - 1):
        subj_v = enc(b, nodes[i])
        obj_v  = enc(b, nodes[i+1])
        b.binding.bind(subj_v, rel_v, obj_v)

def chain_follow(b, start_word, rel_word):
    """Use Op::CHAIN_FOLLOW and return the decoded result word."""
    b.scratchpad.write("subject",  enc(b, start_word), "test")
    b.scratchpad.write("relation", enc(b, rel_word),   "test")
    b.force_reason_step(OP_CHAIN_FOLLOW, "causes")
    result = b.scratchpad.read("result")
    return b.language.best_word(result)


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  Multi-Hop Causal Chain Tests (Op::CHAIN_FOLLOW = 28)")
print("═"*60)

# Test 1: 2-hop chain  A → B → C
def test_2hop():
    b = make_brain()
    bind_chain(b, ["A2", "B2", "C2"], "causes")
    got = chain_follow(b, "A2", "causes")
    check("2-hop: A→B→C, query(A) = C", got == "C2", f"got '{got}'")

# Test 2: 4-hop chain  A → B → C → D → E
def test_4hop():
    b = make_brain()
    bind_chain(b, ["N0", "N1", "N2", "N3", "N4"], "causes")
    got = chain_follow(b, "N0", "causes")
    check("4-hop: N0→...→N4, query(N0) = N4", got == "N4", f"got '{got}'")

# Test 3: 6-hop chain
def test_6hop():
    b = make_brain()
    bind_chain(b, ["X0","X1","X2","X3","X4","X5","X6"], "causes")
    got = chain_follow(b, "X0", "causes")
    check("6-hop: X0→...→X6, query(X0) = X6", got == "X6", f"got '{got}'")

# Test 4: 10-hop chain (max depth)
def test_10hop():
    b = make_brain()
    nodes = [f"Z{i}" for i in range(11)]  # Z0 through Z10
    bind_chain(b, nodes, "causes")
    got = chain_follow(b, "Z0", "causes")
    check("10-hop: Z0→...→Z10, query(Z0) = Z10", got == "Z10", f"got '{got}'")

# Test 5: Cycle detection — A → B → A, should not hang
def test_cycle():
    b = make_brain()
    rel_v = enc(b, "causes")
    a_v = enc(b, "CycA")
    bv  = enc(b, "CycB")
    b.binding.bind(a_v, rel_v, bv)
    b.binding.bind(bv,  rel_v, a_v)  # creates cycle

    import signal
    class TimeoutError(Exception): pass
    def handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(5)  # 5 second timeout
    try:
        got = chain_follow(b, "CycA", "causes")
        signal.alarm(0)
        check("Cycle detection: A→B→A stops safely (no hang)", True)
    except TimeoutError:
        check("Cycle detection: A→B→A stops safely (no hang)", False, "HUNG for 5s!")

# Test 6: Noise resistance — inject irrelevant bindings, chain still found
def test_noise():
    b = make_brain()
    bind_chain(b, ["P0", "P1", "P2", "P3"], "causes")
    # Inject noise: unrelated bindings on different relation
    rel2 = enc(b, "enables")
    for i in range(10):
        n_v = enc(b, f"noise{i}")
        m_v = enc(b, f"noise{i+1}")
        b.binding.bind(n_v, rel2, m_v)
    got = chain_follow(b, "P0", "causes")
    check("Noise resistance: irrelevant bindings don't break chain", got == "P3",
          f"got '{got}'")

# Test 7: Single-hop (1 link) — degrades to BIND_QUERY
def test_1hop():
    b = make_brain()
    bind_chain(b, ["S0", "S1"], "causes")
    got = chain_follow(b, "S0", "causes")
    check("1-hop: S0→S1, query(S0) = S1", got == "S1", f"got '{got}'")

# Test 8: Unknown start node — returns something (no crash)
def test_unknown():
    b = make_brain()
    bind_chain(b, ["Q0", "Q1", "Q2"], "causes")
    # Query a node not in the chain
    b.scratchpad.write("subject",  enc(b, "UNKNOWN_NODE"), "test")
    b.scratchpad.write("relation", enc(b, "causes"),       "test")
    crashed = False
    try:
        b.force_reason_step(OP_CHAIN_FOLLOW, "causes")
    except Exception as e:
        crashed = True
        check("Unknown node: no crash", False, str(e))
    if not crashed:
        check("Unknown node: no crash", True)

# Test 9: Confidence score is written after CHAIN_FOLLOW
def test_confidence_written():
    b = make_brain()
    bind_chain(b, ["C0", "C1", "C2"], "causes")
    chain_follow(b, "C0", "causes")
    conf = b.get_last_confidence()
    check("CHAIN_FOLLOW writes confidence to scratchpad",
          isinstance(conf, float),
          f"conf={conf}")

test_2hop()
test_4hop()
test_6hop()
test_10hop()
test_cycle()
test_noise()
test_1hop()
test_unknown()
test_confidence_written()

# ─────────────────────────────────────────────────────────────────────────────
total  = 9
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
