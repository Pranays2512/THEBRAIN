"""
test_scratchpad.py — Component 11: Scratchpad + Component 12: ReasoningEngine

Scratchpad tests:
  1. write/read round-trip
  2. has() works
  3. read unknown = zero vector
  4. history preserved on overwrite (read_prev)
  5. similarity between two slots
  6. delta = 0 after writing same value twice
  7. stack push/pop/peek
  8. accumulate blends correctly
  9. clear wipes everything

ReasoningEngine tests:
  10. solve_binary("+") produces non-zero result
  11. solve_binary computes same op as Symbolic.apply
  12. infer() chains two steps: A->B->C
  13. loop_until_convergence terminates
  14. ReasoningResult has trace
  15. Brain.scratchpad + Brain.reasoning accessible
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import brain2

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition

def run():
    print("\nScratchpad + ReasoningEngine — Components 11 & 12")
    print("=" * 50)
    results = []

    DIM = 32
    rng = np.random.default_rng(42)

    # ── Scratchpad tests ─────────────────────────────────────────────
    pad = brain2.Scratchpad(n_dims=DIM)

    # 1. write/read
    v = rng.random(DIM).astype(np.float32)
    pad.write("a", v)
    v_back = np.array(pad.read("a"))
    results.append(test("write/read round-trip",
                        float(np.max(np.abs(v - v_back))) < 1e-6))

    # 2. has()
    results.append(test("has() = True after write", pad.has("a")))
    results.append(test("has() = False for unknown", not pad.has("zzz")))

    # 3. read unknown = zero
    z = np.array(pad.read("unknown_slot"))
    results.append(test("read unknown = zero vector",
                        float(np.sum(np.abs(z))) < 1e-6))

    # 4. history on overwrite
    v2 = rng.random(DIM).astype(np.float32)
    pad.write("a", v2)
    prev = np.array(pad.read_prev("a"))
    results.append(test("read_prev returns previous value",
                        float(np.max(np.abs(prev - v))) < 1e-6))

    # 5. similarity
    pad.write("b", v)
    sim = pad.similarity("a", "b")  # a is now v2, b is v
    results.append(test(f"similarity computes cosine ({sim:.3f})",
                        -1.0 <= sim <= 1.0))

    # 6. delta ≈ 0 same value twice
    pad.write("stable", v)
    pad.write("stable", v)  # same value
    d = pad.delta("stable")
    results.append(test(f"delta ≈ 0 for same value (delta={d:.6f})",
                        d < 1e-5))

    # 7. stack push/pop/peek
    pad2 = brain2.Scratchpad(n_dims=DIM)
    s1 = rng.random(DIM).astype(np.float32)
    s2 = rng.random(DIM).astype(np.float32)
    pad2.push(s1)
    pad2.push(s2)
    peek = np.array(pad2.peek())
    popped = np.array(pad2.pop())
    results.append(test("peek = last pushed",
                        float(np.max(np.abs(peek - s2))) < 1e-6))
    results.append(test("pop returns last pushed",
                        float(np.max(np.abs(popped - s2))) < 1e-6))
    results.append(test(f"stack_size after pop = 1", pad2.stack_size == 1))

    # 8. accumulate blends
    pad3 = brain2.Scratchpad(n_dims=DIM)
    ones = np.ones(DIM, dtype=np.float32)
    zeros = np.zeros(DIM, dtype=np.float32)
    pad3.write("x", zeros)
    pad3.accumulate("x", ones, alpha=0.5)
    blended = np.array(pad3.read("x"))
    results.append(test(f"accumulate blends (mean={float(np.mean(blended)):.3f} ≈ 0.5)",
                        abs(float(np.mean(blended)) - 0.5) < 0.01))

    # 9. clear
    pad3.clear()
    results.append(test("clear wipes all slots",
                        pad3.slot_count == 0 and pad3.stack_size == 0))

    # ── ReasoningEngine tests ────────────────────────────────────────
    sym = brain2.Symbolic(n_dims=DIM)
    sym.seed_math_symbols()
    engine = brain2.ReasoningEngine(sym, n_dims=DIM, max_steps=20)

    pad4 = brain2.Scratchpad(n_dims=DIM)
    a = rng.random(DIM).astype(np.float32)
    b = rng.random(DIM).astype(np.float32)

    # 10. solve_binary non-zero
    result_add = np.array(engine.solve_binary("+", a, b, pad4))
    results.append(test(f"solve_binary('+') non-zero (norm={float(np.linalg.norm(result_add)):.4f})",
                        float(np.linalg.norm(result_add)) > 1e-4))

    # 11. solve_binary matches Symbolic.apply
    expected = np.array(sym.apply("+", a, b))
    diff = float(np.max(np.abs(result_add - expected)))
    results.append(test(f"solve_binary matches Symbolic.apply (diff={diff:.6f})",
                        diff < 1e-5))

    # 12. infer() chains A -> B -> C
    A = rng.random(DIM).astype(np.float32)
    B = rng.random(DIM).astype(np.float32)
    C = rng.random(DIM).astype(np.float32)
    pad5 = brain2.Scratchpad(n_dims=DIM)
    chain = [
        brain2.ReasoningStep("A", "+", "B", "AB"),
        brain2.ReasoningStep("AB", "+", "C", "ABC"),
    ]
    premises = [("A", A), ("B", B), ("C", C)]
    conclusion = np.array(engine.infer(premises, chain, pad5))
    results.append(test(f"infer() chains 2 steps (norm={float(np.linalg.norm(conclusion)):.4f})",
                        float(np.linalg.norm(conclusion)) > 1e-4))

    # 13. loop_until_convergence terminates
    pad6 = brain2.Scratchpad(n_dims=DIM)
    pad6.write("x", rng.random(DIM).astype(np.float32))
    step = brain2.ReasoningStep("x", "+", "x", "x")
    loop_result = engine.loop_until_convergence(step, pad6, max_iters=10)
    results.append(test(f"loop_until_convergence terminates (steps={loop_result.steps_taken})",
                        loop_result.steps_taken > 0))

    # 14. ReasoningResult has trace
    pad7 = brain2.Scratchpad(n_dims=DIM)
    pad7.write("p", a)
    pad7.write("q", b)
    steps = [brain2.ReasoningStep("p", "+", "q", "r")]
    res = engine.reason(steps, pad7)
    results.append(test(f"ReasoningResult has trace (len={len(res.trace)})",
                        len(res.trace) > 0))

    # 15. Brain.scratchpad + Brain.reasoning accessible
    br = brain2.Brain(8, 8, 16, hidden_dim=64)
    try:
        sp = br.scratchpad
        re = br.reasoning
        sp.write("test", np.zeros(64, dtype=np.float32))
        results.append(test("Brain.scratchpad + Brain.reasoning accessible", True))
    except Exception as e:
        results.append(test(f"Brain.scratchpad accessible (err: {e})", False))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Components 11 & 12 (Scratchpad + Reasoning): READY\n")
        return True
    else:
        print("Components 11 & 12: NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
