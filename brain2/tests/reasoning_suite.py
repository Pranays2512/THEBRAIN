#!/usr/bin/env python3
"""
reasoning_suite.py — the brain's real report card.

Perplexity asks "does the output match the corpus bit-for-bit?". That is the
wrong test for a system meant to *understand*, not memorize. This suite asks
the right question: given strictly less than the answer, can it DERIVE the
rest, on cases it was never shown? That is the difference between inference
and pattern-matching, and it is exactly where a logic engine can beat an LLM.

Every test stores only PRIMITIVE facts and scores only DERIVED conclusions
(never directly stored). For each correct derivation it also prints the chain
in words — the brain showing its actual work, because the reasoning is an
explicit traversal, not hidden weights.

Tests:
  1. transitive_inference  — store adjacent A>B, B>C; derive A>C (k hops),
                             with DISTRACTOR facts present so retrieval must
                             discriminate, not just walk a clean graph
  2. depth_curve           — accuracy vs hop distance (1=stored .. K=derived)
  3. noise_robustness      — query with perturbed vectors: is it real retrieval
                             or brittle exact-match lookup?
  4. relation_composition  — store parent links; derive ancestor (novel pair)

Pure Python over the built module; no training of the LM required.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

N_DIMS = 64
SEED = 7


class Concepts:
    """Distinct random concept vectors + nearest-neighbor decode."""

    def __init__(self, names, rng):
        self.names = list(names)
        self.vec = {n: rng.standard_normal(N_DIMS).astype(np.float32) for n in self.names}
        M = np.stack([self.vec[n] for n in self.names])
        self.norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    def decode(self, v):
        v = np.asarray(v, dtype=np.float32)
        if v.size == 0 or np.linalg.norm(v) < 1e-8:
            return None, 0.0
        s = self.norm @ (v / np.linalg.norm(v))
        i = int(np.argmax(s))
        return self.names[i], float(s[i])


def make_brain():
    return brain2.Brain(som_rows=32, som_cols=32, n_dims=N_DIMS, hidden_dim=128, seed=SEED)


# ── 1+2. transitive inference + depth curve ──────────────────────────────────
def build_transitive(rng, n_chains=20, chain_len=6, n_distractors=400):
    """Strict-order chains x0>x1>...; store only adjacent edges, plus a pile of
    unrelated distractor facts so the engine can't just assume one global
    chain. Returns (brain, chains, Concepts, rel)."""
    b = make_brain()
    rel = rng.standard_normal(N_DIMS).astype(np.float32)
    chains, all_names = [], []
    for ci in range(n_chains):
        names = [f"c{ci}_{i}" for i in range(chain_len)]
        chains.append(names)
        all_names += names
    distractor_names = [f"d{i}" for i in range(n_distractors * 2)]
    C = Concepts(all_names + distractor_names, rng)
    other_rel = rng.standard_normal(N_DIMS).astype(np.float32)
    for names in chains:
        for i in range(len(names) - 1):
            b.bind_triple(C.vec[names[i]], rel, C.vec[names[i + 1]])
    for i in range(n_distractors):  # noise facts under a different relation
        b.bind_triple(C.vec[f"d{2*i}"], other_rel, C.vec[f"d{2*i+1}"])
    return b, chains, C, rel


def test_transitive(b, chains, C, rel, chain_len=6, verbose_examples=3):
    per_hop, examples = {}, []
    for k in range(1, chain_len):
        correct = total = 0
        for names in chains:
            for i in range(len(names) - k):
                start, gold = names[i], names[i + k]
                vec, _ = b.binding_query(C.vec[start], rel, True, 0.3, k)
                pred, _ = C.decode(vec)
                ok = (pred == gold)
                correct += ok
                total += 1
                if k >= 2 and ok and len(examples) < verbose_examples:
                    chain, cur = [start], start
                    for _ in range(k):
                        v, _ = b.binding_query(C.vec[cur], rel, True, 0.3, 1)
                        nxt, _ = C.decode(v)
                        if nxt is None:
                            break
                        chain.append(nxt)
                        cur = nxt
                    examples.append((start, gold, chain))
        per_hop[k] = correct / max(total, 1)
    derived = [per_hop[k] for k in per_hop if k >= 2]
    return per_hop, (sum(derived) / len(derived) if derived else 0.0), examples


def test_noise(b, chains, C, rel, rng, sigmas=(0.0, 0.2, 0.5, 1.0), hop=3):
    """Derive the hop-distant descendant when the QUERY subject is perturbed by
    Gaussian noise. Exact-match lookup collapses immediately; real similarity
    retrieval degrades gracefully."""
    out = {}
    for s in sigmas:
        correct = total = 0
        for names in chains:
            for i in range(len(names) - hop):
                q = C.vec[names[i]] + rng.standard_normal(N_DIMS).astype(np.float32) * s
                vec, _ = b.binding_query(q.astype(np.float32), rel, True, 0.3, hop)
                pred, _ = C.decode(vec)
                correct += (pred == names[i + hop])
                total += 1
        out[s] = correct / max(total, 1)
    return out


# ── 3. relation composition (grandparent) ────────────────────────────────────
def test_relation_composition(rng, n_families=30):
    """Store (X parent Y). Derive (X ancestor Z) for Z two hops down — a pair
    never stored, requiring composition of the parent relation with itself."""
    b = make_brain()
    parent = rng.standard_normal(N_DIMS).astype(np.float32)
    names, triples = [], []
    for f in range(n_families):
        g, p, c = f"g{f}", f"p{f}", f"c{f}"   # grandparent, parent, child
        names += [g, p, c]
        triples.append((g, p, c))
    C = Concepts(names, rng)
    for g, p, c in triples:
        b.bind_triple(C.vec[g], parent, C.vec[p])
        b.bind_triple(C.vec[p], parent, C.vec[c])

    correct = 0
    for g, p, c in triples:
        vec, _ = b.binding_query(C.vec[g], parent, True, 0.3, 2)  # 2 hops -> child
        pred, _ = C.decode(vec)
        correct += (pred == c)
    return correct / max(len(triples), 1)


def main():
    rng = np.random.default_rng(SEED)
    print("brain2 reasoning report card — derive, don't memorize\n")

    b, chains, C, rel = build_transitive(rng)
    per_hop, derived_acc, examples = test_transitive(b, chains, C, rel)
    print("1. TRANSITIVE INFERENCE (store adjacent only, + 400 distractor facts)")
    print("   depth curve (hop -> accuracy):")
    for k in sorted(per_hop):
        tag = "stored" if k == 1 else "DERIVED"
        print(f"     {k} hop{'s' if k > 1 else ' '} [{tag:7s}]: {per_hop[k]:.2f}")
    print(f"   derived accuracy (>=2 hops, never stored): {derived_acc:.3f}\n")

    print("   derivation read-outs (the brain showing its work):")
    for start, gold, chain in examples:
        print(f"     {start} > {gold}?  derived via  " + " > ".join(chain))
    print()

    noise = test_noise(b, chains, C, rel, rng)
    print("2. NOISE ROBUSTNESS (derive 3-hop with perturbed query; real")
    print("   similarity-retrieval degrades gracefully, exact-match collapses):")
    for s in sorted(noise):
        print(f"     sigma {s:.1f}: {noise[s]:.2f}")
    print()

    comp = test_relation_composition(rng)
    print(f"3. RELATION COMPOSITION (parent∘parent -> ancestor, novel pair): {comp:.3f}\n")

    print("summary:", {
        "transitive_derived_acc": round(derived_acc, 3),
        "noise_robustness_sigma0.5": round(noise.get(0.5, 0.0), 3),
        "relation_composition_acc": round(comp, 3),
    })


if __name__ == "__main__":
    main()
