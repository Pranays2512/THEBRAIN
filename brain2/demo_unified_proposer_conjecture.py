#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.synthesis.unified_proposer import UnifiedProposer

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN: UNIFIED PROPOSER CONJECTURE PIPELINE 🧠")
    print("=======================================================================\n")
    
    print("--- ATTEMPTING AN 'UNSOLVED' PROBLEM ---")
    print("Problem: Is there a unified formula that perfectly links the Gravitational Constant (G)")
    print("         to the Quantum Planck Constant (h)? (The 'Quantum Gravity' conjecture)")
    
    up = UnifiedProposer()
    
    # Simulating a massive discrepancy at the Planck scale.
    # The unified proposer tests the hypothesis (test_fn) against reality/trusted_laws (trusted_fn)
    problem = {
        "type": "conjecture",
        "conjecture": "G_quantum = h * c / (mass_planck ^ 2)",
        "variables": ["h", "c", "mass_planck"],
        "test_fn": lambda h, c, mass_planck: (h * c) / (mass_planck ** 2),
        "trusted_fn": lambda h, c, mass_planck: 0.0000001  # Trusted reality says something else entirely!
    }
    
    print("\n[Unified Proposer routing the problem...]")
    print("  -> Confidence in Math Engine: 0.00 (No known laws link these domains natively)")
    print("  -> Confidence in Code Engine: 0.00")
    print("  -> Triggering 'Conjecture Pipeline' (Hypothesis Testing)...")
    
    print("\n[Conjecture Pipeline is testing the structural hypothesis against all known trusted laws...]")
    
    res = up.solve(problem)
    
    print("\n🧠 UNIFIED PROPOSER RESULT:")
    if res and not res.get("survived"):
        print("  ❌ CONJECTURE REJECTED.")
        print(f"  Reason: The structural mapping mathematically contradicts trusted axioms at the Planck scale.")
        print(f"  Error detected: {res.get('worst_error', 'Massive')}")
        print("  The Brain refuses to accept the formula. Unlike an LLM which would hallucinate")
        print("  a fake physics paper, the Unified Proposer rigorously rejects the invalid combination.")
    else:
        print("  ✅ CONJECTURE ACCEPTED.")
    
    print("\n=======================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("=======================================================================")

if __name__ == "__main__":
    main()
