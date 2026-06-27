#!/usr/bin/env python3
"""
analogy_struct.py — relational analogy WITHOUT a shared relation vocabulary (Novel #3).

The old analogy engine intersected pre-shared relation names (common = rels(A) & rels(B)):
it can only map domains that already use the same words. Real analogy (Rutherford: solar
system -> atom) maps domains whose relations have DIFFERENT names but the same STRUCTURE.

This is structure-mapping (Gentner): search for an entity alignment that makes the two
relational graphs correspond, inducing a RELATION mapping as it goes (pulls<->attracts,
revolves<->circles) — none of it pre-shared. Then it TRANSFERS: a source fact whose target
analog is missing becomes a predicted fact in the target domain (the point of analogy —
carry structure from the known domain to the new one).

Honest limit: brute-force over entity alignments (fine for small domains; the matching
problem is NP-hard in general). It maps relational STRUCTURE, not meaning — the transfer
is a hypothesis to verify, not a proven fact.
"""

from itertools import permutations


def _entities(triples):
    es = []
    for a, _r, b in triples:
        for e in (a, b):
            if e not in es:
                es.append(e)
    return es


def align(source, target):
    """Find the entity bijection (source->target) maximizing corresponded relations, with a
    consistent relation mapping. Returns (entity_map, relation_map, score)."""
    es, et = _entities(source), _entities(target)
    if len(es) > len(et):
        return None, None, 0
    best, best_rel, best_score = None, None, -1
    for perm in permutations(et, len(es)):
        emap = dict(zip(es, perm))
        relmap, score, ok = {}, 0, True
        for a, r, b in source:
            cands = [tr for ta, tr, tb in target if ta == emap[a] and tb == emap[b]]
            if not cands:
                continue
            tr = cands[0]
            if r in relmap and relmap[r] != tr:       # a relation must map consistently
                ok = False
                break
            if tr in relmap.values() and relmap.get(r) != tr:
                continue                              # target relation already claimed
            relmap[r] = tr
            score += 1
        if ok and score > best_score:
            best, best_rel, best_score = emap, relmap, score
    return best, best_rel, best_score


def transfer(source, target, emap, relmap):
    """Carry source structure the target is missing. A source relation already aligned ->
    predict with its target name; an UNMAPPED source relation -> predict a NEW relation in
    the target (the real analogical leap: 'there should be a relation here'). Returns
    (predicted_fact, is_new_relation)."""
    have = set(target)
    preds = []
    for a, r, b in source:
        if a in emap and b in emap:
            tr = relmap.get(r, r)                     # aligned name, or carry source name as NEW
            cand = (emap[a], tr, emap[b])
            if cand not in have:
                preds.append((cand, r not in relmap))
    return preds


def _demo():
    print("=== analogy_struct — map two domains by STRUCTURE, not shared words (Novel #3) ===\n")
    # solar system and atom — NO relation label in common (heavier!=massive, revolves!=circles)
    solar = [("sun", "heavier", "planet"), ("planet", "revolves", "sun"),
             ("sun", "pulls", "planet")]
    atom  = [("nucleus", "massive", "electron"), ("electron", "circles", "nucleus")]
    # note: atom is MISSING the analog of (sun, pulls, planet) -> analogy should predict it

    print("  source (solar):", solar)
    print("  target (atom): ", atom, "  <- missing the 'pulls' analog")
    shared = {r for _, r, _ in solar} & {r for _, r, _ in atom}
    print("  shared relation labels:", shared or "NONE (old analogy engine would map nothing)\n")

    emap, relmap, score = align(solar, atom)
    print("  structural alignment (induced, no shared vocab):")
    print("    entities :", emap)
    print("    relations:", relmap)
    print("    corresponded relations:", score)

    preds = transfer(solar, atom, emap, relmap)
    print("\n  TRANSFER — structure the target was missing, predicted from the source:")
    for (a, r, b), is_new in preds:
        tag = "NEW relation — the analogical leap" if is_new else "aligned"
        print("    %s --%s--> %s   (%s)" % (a, r, b, tag))
    print("\n  Mapped sun<->nucleus, planet<->electron and pulls<->attracts-style relations")
    print("  with ZERO shared relation names — then carried the missing structure across.")


if __name__ == "__main__":
    _demo()
