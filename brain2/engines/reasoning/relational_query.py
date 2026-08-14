#!/usr/bin/env python3
"""
relational_query.py — Phase 4, rung 2: questions RELATING two entities.

Rung 1 answered attribute lookups: "what is the speed of the rocket?". Rung 2
handles STRUCTURE — a comparison between two entities:

    "is the rocket heavier than the probe?"
      -> entities {rocket, probe}, comparative 'heavier' -> relation 'mass',
         direction '>'  -> solve mass for each via the executive, compare.

The comparative is decomposed to its base adjective ("heavier"->"heavy",
"denser"->"dens") and mapped to a relation by REUSING rung 1's matcher (embedding
+ prefix), so synonyms still generalize — "swifter" would route to speed without
a new rule. Direction is read from a small closed class of "less" comparatives.

Each side is computed by the verified executive (facts + policies + proposer), so
"denser" compares a COMPUTED quantity (density = mass/volume), not a stored one.
Honest scope: single binary comparison of one attribute. Multi-clause / boolean
combinations / general relations are the structural-parser + LLM rungs above this.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engines.reasoning.means_ends import Policy
from engines.synthesis.policy_proposer import MultiPolicyMemory, Solver
from engines.knowledge.policy_pack import PACK
from adapters.nl_query import NLQueryParser, load_glove, _rel_words
from engines.reasoning.reasoning_engine import ReasoningEngine

# comparatives that mean "less" (everything else is treated as "more")
LESS = {"lighter", "slower", "smaller", "weaker", "lower", "shorter", "cooler",
        "cheaper", "thinner", "fewer", "less"}
STOP2 = {"is", "the", "than", "a", "an", "more", "which", "or", "has", "does",
        "do", "of", "to", "and", "between"}

# comparative base-adjective -> relation. A small CLOSED class — raw GloVe
# adjective->noun similarity is noisy here (heavy nearest 'pressure', not 'mass'),
# so this lexicon is the honest fix; novel words still fall through to the matcher.
ADJ_RELATION = {
    "heavy": "mass", "light": "mass", "massive": "mass",
    "fast": "speed", "quick": "speed", "swift": "speed", "slow": "speed",
    "dense": "density", "big": "volume", "large": "volume", "small": "volume",
    "strong": "force", "weak": "force",
}


def base_adjective(w):
    """Strip a comparative suffix: heavier->heavy, faster->fast, denser->dens."""
    if w.endswith("ier"):
        return w[:-3] + "y"
    if w.endswith("er"):
        return w[:-2]
    return w


class RelationalParser:
    def __init__(self, parser: NLQueryParser):
        self.p = parser

    def parse(self, sentence):
        """Return (entityX, entityY, relation, direction) or None. direction is
        +1 for 'more' comparisons, -1 for 'less'."""
        toks = re.findall(r"[a-z_]+", sentence.lower())
        ents = [t for t in toks if t in self.p.entities]
        if len(ents) < 2:
            return None
        x, y = ents[0], ents[1]
        # form A: a comparative adjective ("heavier", "denser")
        comp = next((t for t in toks
                     if t.endswith("er") and t not in self.p.entities and t not in STOP2),
                    None)
        if comp is not None:
            direction = -1 if comp in LESS else 1
            base = base_adjective(comp)
            rel = ADJ_RELATION.get(base) or ADJ_RELATION.get(comp)
            if rel is None:                       # novel word -> reuse rung-1 matcher
                rel, _ = self.p.match_relation([base, comp])
            return (x, y, rel, direction) if rel else None
        # form B: "more/less <relation> than" ("more force than")
        if "more" in toks or "less" in toks:
            direction = -1 if "less" in toks else 1
            content = [t for t in toks if t not in self.p.entities
                       and t not in STOP2 and t not in ("more", "less")]
            rel, _ = self.p.match_relation(content)
            return (x, y, rel, direction) if rel else None
        return None

    def answer(self, sentence, kb, mem):
        parsed = self.parse(sentence)
        if parsed is None:
            return None
        x, y, rel, d = parsed
        vx = Solver(kb, mem, use_proposer=True).solve(x, rel)
        vy = Solver(kb, mem, use_proposer=True).solve(y, rel)
        if vx is None or vy is None:
            return f"can't compare — missing {rel} for {x if vx is None else y}"
        more = (vx > vy) if d > 0 else (vx < vy)
        verdict = "Yes" if more else "No"
        comp_phrase = sentence.strip().rstrip("?")
        return (f"{verdict}. {x} {rel}={vx:.4g}, {y} {rel}={vy:.4g}  "
                f"({'>' if vx > vy else '<' if vx < vy else '='})")


def _demo():
    kb = ReasoningEngine()
    bodies = {
        "rocket": {"mass": 1000, "accel": 12, "speed": 300, "volume": 2.0, "molar_mass": 1, "area": 1, "distance": 1},
        "probe":  {"mass": 200,  "accel": 30, "speed": 500, "volume": 0.5, "molar_mass": 1, "area": 1, "distance": 1},
    }
    for ent, fs in bodies.items():
        for r, v in fs.items():
            kb.learn(ent, r, str(v))

    mem = MultiPolicyMemory()
    for t, ins, expr in PACK:
        mem.add(Policy(t, ins, expr))

    entities = set(bodies)
    relations = sorted({t for t, _, _ in PACK} |
                       {r for fs in bodies.values() for r in fs})
    needed = set()
    for r in relations:
        needed.update(_rel_words(r) + [r])
    sentences = [
        "is the rocket heavier than the probe?",
        "is the rocket faster than the probe?",
        "is the probe denser than the rocket?",     # density COMPUTED per entity
        "does the rocket have more force than the probe?",
        "is the probe lighter than the rocket?",
    ]
    for s in sentences:
        needed.update(base_adjective(w) for w in re.findall(r"[a-z_]+", s.lower()))
        needed.update(re.findall(r"[a-z_]+", s.lower()))
    glove = load_glove(needed=needed)

    rp = RelationalParser(NLQueryParser(entities, relations, glove))

    print("=== relational_query — compare two entities (no LLM) ===\n")
    for s in sentences:
        print(f"  > {s}")
        print(f"      {rp.answer(s, kb, mem)}\n")


if __name__ == "__main__":
    _demo()
