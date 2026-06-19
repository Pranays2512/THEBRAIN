#!/usr/bin/env python3
"""
analogy_engine.py — structure mapping between two domains (bounded idea #2).

Analogy aligns RELATIONS, not surface features: a pump maps to a battery because
both *cause flow*, not because they resemble each other. Given two domains as
(subject, relation, object) facts that share a relation vocabulary, this finds an
object correspondence that preserves the relational structure, then TRANSFERS a
source fact whose analog is missing in the target — an analogical prediction.

    water:  pump increases flow,  pipe resists flow,  flow depends_on pressure,
            pump raises pressure
    elec:   battery increases current, resistor resists current,
            current depends_on voltage
    -> mapping pump:battery, flow:current, pipe:resistor, pressure:voltage
    -> predicts:  battery raises voltage     (the analog of 'pump raises pressure')

Honest scope: this is the realistic core of structure mapping — alignment by
relational signature over a SHARED vocabulary, with an unambiguous correspondence.
It is NOT open-ended invention (randomly mapping any domain onto any problem):
the mapping must be determined by the structure, and a transferred fact is a
HYPOTHESIS, not a truth — it should be verified (see inductive_engine) before it
is trusted. Ambiguous or structure-poor domains yield no mapping, honestly.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def _objects(facts):
    return {s for s, _, _ in facts} | {o for _, _, o in facts}


def _relations(facts):
    return {r for _, r, _ in facts}


def _signature(obj, facts, rels):
    """An object's structural role: the (relation, direction) edges it takes part
    in, restricted to the shared relation vocabulary `rels`."""
    sig = set()
    for s, r, o in facts:
        if r not in rels:
            continue
        if s == obj:
            sig.add((r, "subj"))
        if o == obj:
            sig.add((r, "obj"))
    return frozenset(sig)


class AnalogyEngine:
    def map_domains(self, source, target):
        """Return (mapping, transfers). `mapping` is source_obj -> target_obj that
        preserves the shared-relation structure; `transfers` are analogical
        predictions [(t_subj, rel, t_obj, from_source_fact)]."""
        common = _relations(source) & _relations(target)
        s_objs, t_objs = _objects(source), _objects(target)
        s_sig = {o: _signature(o, source, common) for o in s_objs}
        t_sig = {o: _signature(o, target, common) for o in t_objs}

        # candidates: same non-empty signature
        mapping, used = {}, set()
        for so in sorted(s_objs):
            sig = s_sig[so]
            if not sig:
                continue
            cands = [to for to in sorted(t_objs) if t_sig[to] == sig and to not in used]
            if len(cands) == 1:                      # unambiguous correspondence only
                mapping[so] = cands[0]
                used.add(cands[0])

        target_set = set(target)
        transfers = []
        for s, r, o in source:
            if s in mapping and o in mapping:
                cand = (mapping[s], r, mapping[o])
                if cand not in target_set:           # analog missing -> predict it
                    transfers.append((*cand, (s, r, o)))
        return mapping, transfers


def _demo():
    water = [
        ("pump", "increases", "flow"),
        ("pipe", "resists", "flow"),
        ("flow", "depends_on", "pressure"),
        ("pump", "raises", "pressure"),          # the structural fact to transfer
    ]
    elec = [
        ("battery", "increases", "current"),
        ("resistor", "resists", "current"),
        ("current", "depends_on", "voltage"),
    ]
    ae = AnalogyEngine()
    mapping, transfers = ae.map_domains(water, elec)

    print("=== analogy_engine — structure mapping (water -> electricity) ===\n")
    print("Correspondence (by relational role):")
    for s, t in sorted(mapping.items()):
        print(f"  {s:9s} ~ {t}")
    print("\nAnalogical predictions (hypotheses to verify):")
    for ts, r, to, src in transfers:
        print(f"  {ts} {r} {to}     (by analogy with {src[0]} {src[1]} {src[2]})")


if __name__ == "__main__":
    _demo()
