#!/usr/bin/env python3
"""
Test the Whole Pipeline: Perception -> Factual/Compute -> C++ Bridges.
"""
from faculties.whole_brain import WholeBrain

def test_pipeline():
    print("Initializing WholeBrain...")
    brain = WholeBrain()
    
    print("\n--- 1. Testing Code Synthesis Route ---")
    # "write a factorial function" -> code synthesis
    ans = brain.sense("write a factorial function")
    print(ans)
    assert ans["answer"]["kind"] == "code", "Failed to route to code synthesis"
    
    print("\n--- 2. Testing Compute Route & Teaching Bridge ---")
    # "force of the rocket" -> compute -> verified_fact -> learn_from_crisp
    ans = brain.sense("force of the rocket")
    print(ans)
    assert ans["answer"]["kind"] == "compute", "Failed to route to compute"
    assert "1.2e+04" in ans["answer"]["msg"] or "12000" in ans["answer"]["msg"], "Failed to compute force (1000 * 12)"
    
    print("\n--- 3. Testing Factual Route ---")
    # "is a dog an animal"
    brain.kre.learn("dog", "isa", "animal") # inject a quick fact
    ans = brain.sense("is a dog an animal")
    print(ans)
    assert ans["answer"]["kind"] == "factual", "Failed to route to factual"
    assert "Yes" in ans["answer"]["msg"], "Failed to infer isa relation"

    print("\n--- 4. Testing Curiosity Bridge Escalation ---")
    # Force high novelty and consecutive failures
    brain.sense("gibberish that has never been seen before")
    brain.sense("more novel words causing an escalation")
    
    print("\nAll pipeline tests passed.")

if __name__ == "__main__":
    test_pipeline()
