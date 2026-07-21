#!/usr/bin/env python3
"""
test_logical_reasoning.py — Stress testing the logical inference of Brain2.

We will test:
  1. Transitive Logic (ISA chains)
  2. Compositional Logic (A -> B and B -> C implies A -> C through rules)
  3. Contradiction Detection (The membrane rejecting illogical facts)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

def main():
    print("=" * 60)
    print("  Brain2 — Logical Reasoning Test")
    print("=" * 60)
    
    from faculties.whole_brain import WholeBrain
    wb = WholeBrain()
    wb.brain = None # Avoid OMP conflict for pure logical testing

    print("\n--- 1. Transitive Logic (ISA Taxonomy) ---")
    # Add words to the brain's vocabulary whitelist so it doesn't drop them
    wb.concepts.update(["socrates", "human", "mortal", "tom", "sam", "kid", "parent", "grandparent"])
    
    # Feed it custom facts that form a chain
    wb.kre.learn("socrates", "isa", "human")
    wb.kre.learn("human", "isa", "mortal")
    wb.kre.set_transitive("isa")
    
    # Query the brain to see if it infers the logical conclusion
    print("  Fact 1: Socrates is a human")
    print("  Fact 2: Human is mortal")
    result = wb.sense("is socrates mortal?")
    print(f"  > is socrates mortal? -> {result['answer']}")

    print("\n--- 2. Compositional Logic (Rule Chaining) ---")
    wb.kre.learn("tom", "parent", "sam")
    wb.kre.learn("sam", "parent", "kid")
    wb.kre.add_rule("parent", "parent", "grandparent")
    
    # Test if the brain can apply the compositional rule
    print("  Fact 1: Tom is parent of Sam")
    print("  Fact 2: Sam is parent of Kid")
    print("  Rule: parent + parent = grandparent")
    # Using the ReasoningEngine's ask() which applies composition rules
    ans, trace = wb.kre.ask("tom", "grandparent")
    print(f"  > Who is Tom's grandparent target? -> {ans}")
    if ans:
        print(f"    Trace: {trace}")

    print("\n--- 3. Contradiction & Type Checking (The Membrane) ---")
    # Feed a declarative sentence that violates the brain's logical ontology
    # The verb 'eat' requires an animate agent and an organic patient.
    print("  Input: 'the rock ate the car'")
    result = wb.sense("the rock ate the car")
    print(f"  > Result: {result['answer']}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
