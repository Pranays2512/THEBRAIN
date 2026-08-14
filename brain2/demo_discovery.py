#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.synthesis.inductive_engine import InductiveLearner
from engines.reasoning.reasoning_engine import ReasoningEngine

def main():
    print("================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: INVENTION & REDISCOVERY 🧠")
    print("================================================================\n")
    
    print("--- INDUCTIVE LEARNING (Filtering Coincidences) ---")
    print("Feeding raw, messy episodic observations into the Brain...")
    print("  Observed 3 times: [Rain, Wet Ground, Puddles]")
    print("  Observed 2 times: [Cat walked by, Rainbow appeared] (A coincidence!)")
    
    il = InductiveLearner()
    train_obs = [["rain", "wet_ground", "puddles"]] * 3 + [["cat", "rainbow"]] * 2
    test_obs = [["rain", "wet_ground", "puddles"]] * 2 + [["cat", "wind"], ["cat", "cloud"]]
    
    print("\n[Brain is mining inductive rules from observations...]")
    promoted, rejected = il.mine(train_obs, test_obs)
    
    print("\n🛡️ Spurious Rules Rejected:")
    for a, b, why in rejected:
        print(f"  • {a} -> {b} (Rejected: {why})")
        
    print("\n✅ Verified Rules Promoted to Deep Knowledge:")
    for r in promoted:
        print(f"  • {r.a} -> {r.b} (Confidence: {r.conf_test * 100:.0f}%)")
        
    print("\n[Brain reasoning with new inductive rules...]")
    re = il.promote_into(ReasoningEngine(), promoted)
    reaches, _ = re.reaches("rain", "leads_to", "puddles")
    print(f"  User: Does rain lead to puddles?")
    print(f"  Brain: {'Yes' if reaches else 'No'}")
    
    print("\n================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    main()
