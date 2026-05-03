"""
test_symbolic.py — Component 10: Symbolic Binding unit tests

Tests:
  1. bind() registers symbol, knows() returns True
  2. lookup() returns n_dims vector for known symbol
  3. lookup() returns zero vector for unknown symbol
  4. Symbol vectors are stable (lookup twice = same result)
  5. Two different symbols have different vectors
  6. apply("+") produces non-zero vector from two concept vecs
  7. apply("=") returns high similarity when vectors are similar
  8. nearest_symbol() round-trip: lookup vec → nearest = original symbol
  9. seed_math_symbols() registers standard math set
  10. Save/load preserves symbol vectors
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tempfile
import brain2

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition

def run():
    print("\nSymbolic — Component 10")
    print("=" * 50)
    results = []

    DIM = 32
    sym = brain2.Symbolic(n_dims=DIM)

    # 1. bind() + knows()
    sym.bind("plus", op=brain2.SymbolOp.ADD, category="math")
    results.append(test("bind() + knows() = True", sym.knows("plus")))
    results.append(test("unknown symbol knows() = False", not sym.knows("xyzzy")))

    # 2. lookup known symbol → n_dims vector
    vec = np.array(sym.lookup("plus"))
    results.append(test(f"lookup known = {DIM}d vector (got {len(vec)})", len(vec) == DIM))

    # 3. lookup unknown → zero vector
    zero = np.array(sym.lookup("xyzzy"))
    results.append(test("lookup unknown = zero vector",
                        float(np.sum(np.abs(zero))) < 1e-6))

    # 4. Stable: lookup twice same
    v1 = np.array(sym.lookup("plus"))
    v2 = np.array(sym.lookup("plus"))
    results.append(test("Symbol vector stable (lookup twice)",
                        float(np.max(np.abs(v1 - v2))) < 1e-6))

    # 5. Different symbols → different vectors
    sym.bind("minus", op=brain2.SymbolOp.SUBTRACT, category="math")
    v_plus  = np.array(sym.lookup("plus"))
    v_minus = np.array(sym.lookup("minus"))
    diff = float(np.max(np.abs(v_plus - v_minus)))
    results.append(test(f"Different symbols → different vectors (diff={diff:.4f})",
                        diff > 0.01))

    # 6. apply("+") non-zero
    rng = np.random.default_rng(0)
    a = rng.random(DIM).astype(np.float32)
    b = rng.random(DIM).astype(np.float32)
    sym2 = brain2.Symbolic(n_dims=DIM)
    sym2.seed_math_symbols()
    result_add = np.array(sym2.apply("+", a, b))
    results.append(test(f"apply('+') non-zero (norm={float(np.linalg.norm(result_add)):.4f})",
                        float(np.linalg.norm(result_add)) > 1e-4))

    # 7. apply("=") high similarity for similar vectors
    v_similar = a + rng.random(DIM).astype(np.float32) * 0.01
    result_eq = np.array(sym2.apply("=", a, v_similar))
    result_diff = np.array(sym2.apply("=", a, -a))  # opposite
    mean_eq   = float(np.mean(result_eq))
    mean_diff = float(np.mean(result_diff))
    results.append(test(f"apply('=') similar>{mean_eq:.3f} > opposite={mean_diff:.3f}",
                        mean_eq > mean_diff))

    # 8. nearest_symbol round-trip
    sym3 = brain2.Symbolic(n_dims=DIM)
    sym3.seed_math_symbols()
    pi_vec = np.array(sym3.lookup("pi"))
    nearest = sym3.nearest_symbol(pi_vec)
    results.append(test(f"nearest_symbol round-trip: pi→'{nearest}'",
                        nearest == "pi"))

    # 9. seed_math_symbols registers standard set
    sym4 = brain2.Symbolic(n_dims=DIM)
    sym4.seed_math_symbols()
    required = ["+", "-", "*", "=", "pi", "0", "1"]
    all_known = all(sym4.knows(s) for s in required)
    results.append(test(f"seed_math_symbols registers {sym4.symbol_count} symbols",
                        all_known and sym4.symbol_count >= 7))

    # 10. Save/load
    sym5 = brain2.Symbolic(n_dims=DIM)
    sym5.seed_math_symbols()
    v_before = np.array(sym5.lookup("+"))
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    sym5.save(path)
    sym5_loaded = brain2.Symbolic.load(path)
    os.unlink(path)
    v_after = np.array(sym5_loaded.lookup("+"))
    diff = float(np.max(np.abs(v_before - v_after)))
    results.append(test(f"Save/load preserves symbol vectors (diff={diff:.6f})",
                        diff < 1e-5))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 10 (Symbolic): READY\n")
        return True
    else:
        print("Component 10 (Symbolic): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
