#!/usr/bin/env python3
"""
world_knowledge.py — load basic human knowledge of the world (ConceptNet).

ConceptNet 5.7 is a crowd-built common-sense knowledge graph — already
(subject, relation, object) triples, which is exactly what the binding memory
eats. This extracts a CURATED English subset (high-weight, common relations)
into the brain's relation vocabulary, and caches it so re-runs are instant.

Honest scope: ConceptNet is real common sense but noisy and incomplete; this is
the "feed it curated, trusted knowledge" path, not "crawl the web." A few
thousand high-weight assertions make a responsive, demonstrable world model.

    facts = load_conceptnet(max_facts=6000)   # [(subj, rel, obj), ...]
"""

import gzip
import json
import os

def _brain2_root():
    """brain2/ root — walk up from this module (survives being inside a package)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != "/" and not os.path.exists(os.path.join(d, "brain2.cpp")):
        d = os.path.dirname(d)
    return d if os.path.exists(os.path.join(d, "brain2.cpp")) else \
        os.path.dirname(os.path.abspath(__file__))

HERE = _brain2_root()
CN_GZ = os.path.join(HERE, "train", "conceptnet-assertions-5.7.0.csv.gz")
CACHE = os.path.join(HERE, "train", "conceptnet_en_subset.json")

# ConceptNet relation -> brain relation. The clean, reasoning-useful ones
# (IsA is transitive; the rest describe properties/parts/abilities).
REL_MAP = {
    "IsA": "isa", "HasProperty": "is", "PartOf": "part_of", "HasA": "has",
    "UsedFor": "used_for", "CapableOf": "can", "MadeOf": "made_of",
}
# Everyday concepts + their category ancestors, so recognizable multi-hop
# chains exist (dog -> mammal -> animal). We collect every assertion whose
# SUBJECT is one of these; including the ancestors keeps the chains connected.
SEEDS = set("""
dog cat bird fish horse cow pig sheep lion tiger bear mouse rabbit snake frog
mammal reptile amphibian insect animal pet vertebrate organism creature
apple banana orange grape lemon cherry strawberry potato carrot tomato onion
fruit vegetable food plant tree flower grass leaf
car bus truck train plane boat ship bicycle vehicle
house building room kitchen door window wall roof chair table bed sofa furniture
book pen pencil paper computer phone clock knife spoon fork cup glass bottle tool
water fire air sun moon star rain snow cloud river ocean mountain rock metal wood
person human child man woman teacher doctor
""".split())

# concrete categories we want chains to bottom out in (kept even if long-ish)
ANCESTORS = {"animal", "mammal", "plant", "fruit", "vegetable", "food", "vehicle",
             "furniture", "tool", "organism", "living_thing", "object", "device",
             "machine", "substance", "material"}


def _concept(uri):
    """/c/en/dog/n -> 'dog' (English only, drop POS)."""
    p = uri.split("/")
    return p[3] if len(p) >= 4 and p[2] == "en" else None


def extract(min_weight=2.0):
    keep = SEEDS | ANCESTORS
    out, seen = [], set()
    with gzip.open(CN_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            if "/c/en/" not in line:                  # cheap reject
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            brel = REL_MAP.get(cols[1].rsplit("/", 1)[-1])
            if brel is None:
                continue
            s, o = _concept(cols[2]), _concept(cols[3])
            if not s or not o or s == o or s not in keep or len(o) > 25:
                continue
            try:
                if json.loads(cols[4]).get("weight", 1.0) < min_weight:
                    continue
            except Exception:
                continue
            t = (s, brel, o)
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def load_conceptnet(min_weight=2.0, refresh=False):
    if not refresh and os.path.exists(CACHE):
        with open(CACHE) as f:
            return [tuple(t) for t in json.load(f)]
    facts = extract(min_weight)
    with open(CACHE, "w") as f:
        json.dump(facts, f)
    return facts


if __name__ == "__main__":
    import time
    t = time.time()
    facts = load_conceptnet(refresh=True)
    ents = {s for s, _, o in facts} | {o for _, _, o in facts}
    rels = {}
    for _, r, _ in facts:
        rels[r] = rels.get(r, 0) + 1
    print(f"loaded {len(facts)} facts, {len(ents)} concepts, in {time.time()-t:.0f}s")
    print(f"relations: {rels}")
    print("sample facts:")
    for s, r, o in facts[:12]:
        print(f"    {s} {r} {o}")
