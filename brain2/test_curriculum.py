#!/usr/bin/env python3
"""
test_curriculum.py — Tests the Brain's reasoning over the newly ingested CS/Science knowledge.

It loads the data/brain_curriculum.txt file into the ReasoningEngine, 
runs transitive closure (so the Brain deduces facts that were implied but not explicitly stated),
and then asks questions to prove it learned the concepts.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.reasoning.reasoning_engine import ReasoningEngine

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "brain_curriculum.txt")

def main():
    print("==========================================================")
    print("  Brain2 Curriculum Test (Testing Acquired Knowledge)")
    print("==========================================================")
    
    kre = ReasoningEngine()
    
    # 1. Load the knowledge
    print("[-] Loading knowledge from data/brain_curriculum.txt...")
    facts_loaded = 0
    with open(DATA_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Format: FACT: s | r | o   or   ISA: s | o
            if line.startswith("FACT:"):
                parts = line[5:].split("|")
                if len(parts) == 3:
                    s, r, o = [p.strip() for p in parts]
                    kre.learn(s, r, o)
                    facts_loaded += 1
            elif line.startswith("ISA:"):
                parts = line[4:].split("|")
                if len(parts) == 2:
                    s, o = [p.strip() for p in parts]
                    kre.learn(s, "isa", o)
                    facts_loaded += 1
                    
    print(f"[+] Loaded {facts_loaded} strict logical triples into the Reasoning Engine.")
    
    # 2. Run transitive closure (derive unstated facts)
    print("[-] Running Type Closure (deducing multi-hop inheritance)...")
    kre.set_transitive("isa")
    
    # 3. Test the knowledge!
    print("\n[+] Testing concepts from the curriculum:\n")
    
    tests = [
        # Explicit facts it extracted
        ("microsoft", "made", "?x"),
        ("middleware", "part_of", "?x"),
        ("unikernel", "deployed_to", "?x"),
        # Calculus Facts
        ("calculus", "studies", "?x"),
        ("differential_calculus", "studies", "?x"),
        ("calculus", "provides", "?x"),
        ("calculus", "used_for", "?x"),
        # Networking Facts
        ("george_stibitz", "created", "?x"),
        ("network", "part_of", "?x"),
        # Chemistry Facts
        ("carbon", "has", "?x"),
        ("organic_compounds", "form", "?x"),
        # Biology Facts
        ("cytopathology", "used_for", "?x"),
    ]
    
    for s, r, o in tests:
        ans = kre.ask(s, r)
        print(f"  Q: What is '{s} {r}'? \n  A: {ans}")
        
    print("\n[+] Testing Inverse Queries (finding subjects by object):\n")
    
    # Finding who made windows
    for (s, r, obj) in kre.kb.facts:
        if r == "made" and obj == "windows":
            print(f"  Q: Who 'made windows'? \n  A: {s}")
            
    # Finding what is part of an operating system
    for (s, r, obj) in kre.kb.facts:
        if r == "part_of" and obj == "operating_system":
            print(f"  Q: What is 'part_of operating_system'? \n  A: {s}")

if __name__ == "__main__":
    main()
