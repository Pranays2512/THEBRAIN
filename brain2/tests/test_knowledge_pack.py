#!/usr/bin/env python3
"""
test_knowledge_pack.py — generic loaders (TSV / N-Triples) and the fused pack.

Pins that structured dumps ingest through the format loaders, dedupe across
sources, and that build_pack fuses curated knowledge into one store the brain
covers.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledge_base import KnowledgeBase
from knowledge_pack import build_pack, _coverage

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nKnowledgePack — loaders + fused pack")

    with tempfile.TemporaryDirectory() as d:
        # TSV loader
        tsv = os.path.join(d, "facts.tsv")
        with open(tsv, "w") as f:
            f.write("Mercury\tisa\tplanet\n")
            f.write("Venus\tisa\tplanet\n")
            f.write("bad line without tabs\n")          # skipped
        kb = KnowledgeBase()
        n = kb.ingest_tsv(tsv)
        check("TSV loads well-formed rows, skips junk", n == 2)
        check("TSV facts normalized", ("mercury", "isa", "planet") in kb.facts)
        check("TSV tracks source", kb.by_source.get("facts.tsv") == 2)

        # N-Triples loader (local names from URIs)
        nt = os.path.join(d, "data.nt")
        with open(nt, "w") as f:
            f.write("<http://ex.org/Paris> <http://ex.org/capitalOf> <http://ex.org/France> .\n")
            f.write("<http://ex.org/Berlin> <http://ex.org/capitalOf> <http://ex.org/Germany> .\n")
        kb2 = KnowledgeBase()
        n2 = kb2.ingest_ntriples(nt)
        check("N-Triples loads local names", n2 == 2)
        check("N-Triples strips URI to local name",
              ("paris", "capitalof", "france") in kb2.facts)

        # build_pack fuses core + a TSV (skip ConceptNet for speed)
        pack = build_pack(extra_tsv=[tsv], conceptnet=False)
        check("pack includes the core seed", ("dog", "isa", "mammal") in pack.facts)
        check("pack includes the extra TSV", ("mercury", "isa", "planet") in pack.facts)

        # round-trip the pack
        p = os.path.join(d, "pack.json")
        pack.save(p)
        reloaded = KnowledgeBase.load(p)
        check("pack round-trips", reloaded.facts == pack.facts)

    # coverage on the fused (core-only here) pack
    a, t = _coverage(build_pack(conceptnet=False), ["what is a dog?", "what is a unicorn?"])
    check("brain covers a seeded entity, not an unseeded one", a == 1 and t == 2)

    print(f"\nKnowledge pack: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
