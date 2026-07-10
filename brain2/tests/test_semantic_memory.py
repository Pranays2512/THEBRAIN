#!/usr/bin/env python3
"""
test_semantic_memory.py — the memory generalizing via meaning.

Pins the capability a dict cannot have: answer a query about a word never
stored, because it is semantically near a word that was — and DON'T answer for
unrelated words (proving it is genuine similarity, not match-everything).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.knowledge.semantic_memory import SemanticMemory, SemanticError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nSemanticMemory — generalize via meaning")
    sm = SemanticMemory()
    sm.learn("automobile", "has", "engine")
    sm.learn("dog", "has", "tail")
    sm.learn("doctor", "treats", "patients")

    # 1. exact retrieval
    o, c = sm.ask("automobile", "has")
    check("exact retrieval works", o == "engine" and c > 0.9)

    # 2. THE capability: generalize to a never-stored synonym
    o, _ = sm.ask("car", "has")             # "car" never stored; car ~ automobile
    check("generalizes car -> automobile's fact", o == "engine")
    o, _ = sm.ask("puppy", "has")           # puppy ~ dog
    check("generalizes puppy -> dog's fact", o == "tail")
    o, _ = sm.ask("physician", "treats")    # physician ~ doctor
    check("generalizes physician -> doctor's fact", o == "patients")

    # 3. genuine similarity: an UNRELATED word does NOT match (not a dict, not
    #    match-everything)
    o, c = sm.ask("apple", "has")           # apple unrelated to car/dog
    check("unrelated word -> no false match", o is None)

    # 4. similar(): semantically-closest known token first (a dict can't)
    check("similar('car') ranks automobile first", sm.similar("car", 1) == ["automobile"])

    # 5. it really is the embeddings: a coined OOV word has no neighbours
    o, _ = sm.ask("zqxwv", "has")           # not a real word -> random vec
    check("coined word does not match", o is None)

    # 6. idempotency + validation
    check("idempotent learn", sm.learn("automobile", "has", "engine") is False)
    for bad in ("", "  ", None):
        try:
            sm.learn(bad, "has", "x"); check(f"reject {bad!r}", False)
        except SemanticError:
            check(f"reject {bad!r}", True)

    print(f"\nSemantic memory: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
