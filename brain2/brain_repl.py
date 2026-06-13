#!/usr/bin/env python3
"""
brain_repl.py — teach it facts, ask it something you never stated.

The demo for what brain2 actually is: a CPU-native system that learns facts
online (no retraining), derives conclusions it was never told, and explains
its reasoning by reading back the actual chain — because the reasoning is an
explicit traversal of its binding memory, not hidden weights.

Usage:
    python3 brain_repl.py            # interactive
    python3 brain_repl.py --demo     # scripted walkthrough

Interactive grammar (entities and relations are single tokens):
    alice > bob            teach a fact            (alice  >  bob)
    tom parent sam         teach with any relation
    alice > ?              ask: what does alice relate to (transitively)?
    alice > emma           ask yes/no, with derivation
    facts                  list everything taught
    help / quit
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

N_DIMS = 64
MAX_DEPTH = 8


class FactBrain:
    def __init__(self):
        self.b = brain2.Brain(som_rows=32, som_cols=32, n_dims=N_DIMS, hidden_dim=128, seed=1)
        self.vecs = {}            # token -> concept vector (deterministic)
        self.entities = set()     # tokens used as subject/object
        self.relations = set()
        self.facts = []           # (subj, rel, obj) for display

    def vec(self, token):
        if token not in self.vecs:
            h = abs(hash(token)) % (2**32)
            self.vecs[token] = np.random.default_rng(h).standard_normal(N_DIMS).astype(np.float32)
        return self.vecs[token]

    def teach(self, subj, rel, obj):
        self.b.bind_triple(self.vec(subj), self.vec(rel), self.vec(obj))
        self.entities.update((subj, obj))
        self.relations.add(rel)
        self.facts.append((subj, rel, obj))

    def _decode(self, v):
        v = np.asarray(v, dtype=np.float32)
        if v.size == 0 or np.linalg.norm(v) < 1e-8 or not self.entities:
            return None, 0.0
        names = sorted(self.entities)
        M = np.stack([self.vec(n) for n in names])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        s = M @ (v / np.linalg.norm(v))
        i = int(np.argmax(s))
        return names[i], float(s[i])

    def hop(self, subj, rel):
        """Single transitive step subj --rel--> ? (depth=1)."""
        vec, conf = self.b.binding_query(self.vec(subj), self.vec(rel), True, 0.3, 1)
        return self._decode(vec)

    def chain(self, subj, rel, target=None):
        """Walk single hops to reconstruct the derivation path."""
        path = [subj]
        cur = subj
        seen = {subj}
        for _ in range(MAX_DEPTH):
            nxt, conf = self.hop(cur, rel)
            if nxt is None or nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            cur = nxt
            if target is not None and nxt == target:
                break
        return path

    def ask_endpoint(self, subj, rel):
        path = self.chain(subj, rel)
        end = path[-1] if len(path) > 1 else None
        return end, path

    def ask_yesno(self, subj, rel, obj):
        path = self.chain(subj, rel, target=obj)
        return (obj in path and obj != subj), path


def show_derivation(rel, path):
    arrow = "  " + f"  {rel}  ".join(path)
    steps = ", ".join(f"{a} {rel} {b}" for a, b in zip(path, path[1:]))
    return f"{arrow}\n  (chained: {steps})"


def handle(fb, line):
    line = line.strip()
    if not line:
        return True
    low = line.lower()
    if low in ("quit", "exit", "q"):
        return False
    if low in ("help", "?"):
        print(__doc__.split("Interactive grammar")[1])
        return True
    if low == "facts":
        if not fb.facts:
            print("  (nothing taught yet)")
        for s, r, o in fb.facts:
            print(f"  {s} {r} {o}")
        return True

    toks = line.split()
    if len(toks) != 3:
        print("  ? expected 'A rel B'  or  'A rel ?'   (try: help)")
        return True
    subj, rel, obj = toks

    if obj == "?":                                  # query: what does A relate to?
        end, path = fb.ask_endpoint(subj, rel)
        if end is None:
            print(f"  I don't know what {subj} {rel} anything.")
        elif len(path) == 2:
            print(f"  {subj} {rel} {end}  (told directly)")
        else:
            print(f"  {subj} {rel} {end}   — DERIVED, never told. Here's how:")
            print(show_derivation(rel, path))
        return True

    # statement OR yes/no question. If both subj & obj already known, treat
    # as a question; otherwise teach. (A leading '?' or trailing '?' also asks.)
    asking = subj in fb.entities and obj in fb.entities and \
        any(fb.facts) and not (subj, rel, obj) in fb.facts
    if asking:
        yes, path = fb.ask_yesno(subj, rel, obj)
        if yes and len(path) == 2:
            print(f"  yes — {subj} {rel} {obj} (told directly)")
        elif yes:
            print(f"  yes — DERIVED, never told. Here's how:")
            print(show_derivation(rel, path))
        else:
            print(f"  not that I can derive.")
    else:
        fb.teach(subj, rel, obj)
        print(f"  ok: {subj} {rel} {obj}")
    return True


DEMO = [
    ("# Teach a ranking chain, one link at a time", None),
    ("alice > bob", "teach"),
    ("bob > carol", "teach"),
    ("carol > dave", "teach"),
    ("dave > emma", "teach"),
    ("# Now ask something it was NEVER told:", None),
    ("alice > ?", "ask"),
    ("alice > emma", "ask"),
    ("# Online learning: add a new link mid-conversation", None),
    ("emma > frank", "teach"),
    ("alice > ?", "ask"),
    ("# A different relation entirely", None),
    ("tom parent sam", "teach"),
    ("sam parent kid", "teach"),
    ("tom parent ?", "ask"),
]


def run_demo():
    fb = FactBrain()
    print("=== brain2 — learn online, derive the unstated, explain the steps ===\n")
    for line, kind in DEMO:
        if kind is None:
            print(f"\n{line}")
            continue
        print(f"> {line}")
        handle(fb, line)


def main():
    if "--demo" in sys.argv:
        run_demo()
        return
    fb = FactBrain()
    print("brain2 fact REPL. Teach 'A rel B', ask 'A rel ?'. 'help' or 'quit'.\n")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not handle(fb, line):
            break


if __name__ == "__main__":
    main()
