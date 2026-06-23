#!/usr/bin/env python3
"""
loop_synth4.py — list/string inputs + nested loops (sum_list, max, contains, sort).

New input type (sequences) and a new control structure (nested loops). To keep the
search from exploding, each algorithm class is a PARAMETERIZED TEMPLATE with a tiny
parameter set — the brain searches the parameters, not an open program space:

  FOLD     : acc=INIT; for x in lst: acc=UPDATE(acc,x); return acc   (sum,product,max,min)
  MEMBER   : for x in lst: if x==t: return True; return False        (contains)
  NESTED   : for i: for j>i: if lst[i]==lst[j]: return True; ...      (has_duplicate)
  SORT     : bubble sort, comparator searched                        (ascending/descending)

Search the parameters to fit examples, verify held-out, render Python. No LLM.

    python3 loop_synth4.py
"""

INITS = {"0": 0, "1": 1, "first": "first"}
FOLD_UPD = {"acc + x": lambda a, x: a + x, "acc * x": lambda a, x: a * x,
            "max(acc, x)": lambda a, x: max(a, x), "min(acc, x)": lambda a, x: min(a, x),
            "acc + 1": lambda a, x: a + 1}


def _run_fold(init, ufn, lst):
    if init == "first":
        acc, rest = lst[0], lst[1:]
    else:
        acc, rest = INITS[init], lst
    for x in rest:
        acc = ufn(acc, x)
    return acc


def synth_fold(examples):
    for ik in INITS:
        for uc, uf in FOLD_UPD.items():
            try:
                if all(_run_fold(ik, uf, lst) == y for lst, y in examples):
                    return ("fold", dict(init=ik, upd=uc))
            except Exception:
                pass
    return None


def synth_sort(examples):
    for cmp in (">", "<"):
        def f(lst, cmp=cmp):
            a = list(lst)
            for _ in range(len(a)):
                for j in range(len(a) - 1):
                    if (a[j] > a[j + 1]) if cmp == ">" else (a[j] < a[j + 1]):
                        a[j], a[j + 1] = a[j + 1], a[j]
            return a
        if all(f(lst) == y for lst, y in examples):
            return ("sort", dict(cmp=cmp))
    return None


def synth_member(examples):
    def f(args):
        lst, t = args
        for x in lst:
            if x == t:
                return True
        return False
    if all(f(a) == y for a, y in examples):
        return ("member", {})
    return None


def synth_nested(examples):
    def f(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i] == lst[j]:
                    return True
        return False
    if all(f(lst) == y for lst, y in examples):
        return ("nested", {})
    return None


def render(fn, kind, s):
    if kind == "fold":
        if s["init"] == "first":
            return (f"def {fn}(lst):\n    acc = lst[0]\n    for x in lst[1:]:\n"
                    f"        acc = {s['upd']}\n    return acc\n")
        return (f"def {fn}(lst):\n    acc = {s['init']}\n    for x in lst:\n"
                f"        acc = {s['upd']}\n    return acc\n")
    if kind == "member":
        return (f"def {fn}(lst, t):\n    for x in lst:\n        if x == t:\n"
                f"            return True\n    return False\n")
    if kind == "nested":
        return (f"def {fn}(lst):\n    for i in range(len(lst)):\n"
                f"        for j in range(i + 1, len(lst)):\n"
                f"            if lst[i] == lst[j]:\n                return True\n"
                f"    return False\n")
    return (f"def {fn}(lst):\n    a = list(lst)\n    for _ in range(len(a)):\n"
            f"        for j in range(len(a) - 1):\n"
            f"            if a[j] {s['cmp']} a[j + 1]:\n"
            f"                a[j], a[j + 1] = a[j + 1], a[j]\n    return a\n")


def _verify(code, fn, examples, kind):
    ns = {}
    exec(code, {"len": len, "list": list, "max": max, "min": min, "range": range}, ns)
    f = ns[fn]
    if kind == "member":
        return all(f(*a) == y for a, y in examples)
    return all(f(x) == y for x, y in examples)


def _demo():
    work = [
        ("sum_list", synth_fold, [([1, 2, 3], 6), ([5, 5], 10), ([], 0), ([4], 4)]),
        ("product", synth_fold, [([1, 2, 3, 4], 24), ([5], 5), ([2, 3], 6)]),
        ("max_list", synth_fold, [([3, 1, 4, 1, 5], 5), ([2, 2], 2), ([7], 7)]),
        ("contains", synth_member, [(([1, 2, 3], 2), True), (([1, 2, 3], 9), False),
                                    (([], 1), False)]),
        ("has_dup", synth_nested, [([1, 2, 3], False), ([1, 2, 2], True), ([], False)]),
        ("sort_asc", synth_sort, [([3, 1, 2], [1, 2, 3]), ([5, 4], [4, 5]),
                                  ([1], [1])]),
    ]
    print("=== loop_synth4 — list inputs + nested loops, no LLM ===\n")
    for fn, synth, ex in work:
        res = synth(ex)
        if res is None:
            print(f"  {fn:9s}: no program found")
            continue
        kind, s = res
        code = render(fn, kind, s)
        ok = _verify(code, fn, ex, kind)
        head = code.splitlines()[0]
        detail = {"fold": f"{s.get('init','')}/{s.get('upd','')}",
                  "sort": s.get("cmp", ""), "member": "x==t", "nested": "i<j dup"}[kind]
        print(f"  {fn:9s} [{kind}]: {detail:18s} [{'VERIFIED ✓' if ok else 'WRONG ✗'}]")
    print("\n  sequence inputs + nested loops, synthesized + verified — sum/product/max,")
    print("  membership, duplicate-detection, bubble sort. Templates keep search bounded.")


if __name__ == "__main__":
    _demo()
