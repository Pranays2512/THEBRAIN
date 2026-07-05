#!/usr/bin/env python3
"""exam_math.py — does the brain COMPUTE the curriculum's arithmetic?

The critique was right: the old exam tested zero math. The curriculum's math
files are full of arithmetic identities — "LAW: 9 + 1 = 10", "LAW: 6 * 3 = 18",
"LAW: 20 - 14 = 6" — that KD dropped (their LHS isn't a variable name, so they
were never admitted as policies). But they ARE checkable computations.

This exam pulls every such identity from math1-8, evaluates the LEFT side with
the brain's LEARNED arithmetic (math_synth, grounded in succ/pred — no host
+ - * /), and checks it equals the RIGHT side. A database can't fake this: the
answer is a computed value, and every op runs on a procedure the brain LEARNED.

    /opt/homebrew/bin/python3.13 exam_math.py
"""
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import knowledge_distill as KD

MATH_FILES = [f"data/math{i}.txt" for i in range(1, 10)]

_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def numeric_leaves_only(tree):
    """True if the expression tree has NO variables — pure arithmetic."""
    if isinstance(tree, str):
        return False
    if not isinstance(tree, tuple):
        return True
    return all(numeric_leaves_only(x) for x in tree[1:])


def collect_identities(paths):
    """Yield (expr_str, lhs_tree, rhs_value) for every 'LAW: <expr> = <number>'."""
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln.startswith("LAW:"):
                    continue
                body = ln[4:].strip()
                if "=" not in body:
                    continue
                if "√" in body or "∠" in body or "∞" in body:
                    continue                # irrational / geometric — not integer arithmetic
                lhs, rhs = (x.strip() for x in body.rsplit("=", 1))
                if not _NUM.match(rhs):
                    continue
                tree = KD.infix_to_tree(lhs)
                if tree is None or not numeric_leaves_only(tree):
                    continue
                out.append((lhs, tree, float(rhs)))
    return out


def main():
    print("=" * 64)
    print("  MATH EXAM — brain computes curriculum arithmetic (LEARNED)")
    print("=" * 64)
    items = collect_identities(MATH_FILES)
    print(f"\nCollected {len(items)} arithmetic identities from math1-8.\n")

    KD.reset_arith_stats()
    passed = failed = 0
    fails = []
    for expr, tree, rhs in items:
        try:
            got = KD._eval(tree, {})           # every +,-,* runs on learned procs
        except Exception as e:
            got = None
        ok = got is not None and abs(got - rhs) < 1e-9
        if ok:
            passed += 1
        else:
            failed += 1
            fails.append((expr, rhs, got))

    # sample of what it solved, by operator
    by_op = {"+": [], "-": [], "*": []}
    for expr, tree, rhs in items:
        op = tree[0] if isinstance(tree, tuple) else "?"
        if op in by_op and len(by_op[op]) < 3:
            try:
                by_op[op].append(f"{expr} = {KD._eval(tree, {}):g}")
            except Exception:
                pass
    print("  sample solved (computed, not looked up):")
    for op, egs in by_op.items():
        for e in egs:
            print(f"    {e}")

    if fails:
        print(f"\n  failures ({len(fails)}):")
        for expr, rhs, got in fails[:12]:
            print(f"    {expr} = {rhs}  but brain got {got}")

    st = KD.ARITH_STATS
    total_ops = st["learned"] + st["host"]
    pct_learned = 100 * st["learned"] // total_ops if total_ops else 0
    print("\n" + "=" * 64)
    print(f"  SCORE: {passed}/{len(items)}  ({100*passed//len(items) if items else 0}%)")
    print(f"  arithmetic ops executed: {st['learned']} LEARNED + {st['host']} host "
          f"= {pct_learned}% on learned procedures")
    print(f"  (host = division / floats / operands too big for grounded recursion)")
    print("=" * 64)


if __name__ == "__main__":
    main()
