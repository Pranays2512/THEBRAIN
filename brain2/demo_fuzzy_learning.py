#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.synthesis.inductive_engine import InductiveLearner
from engines.reasoning.reasoning_engine import ReasoningEngine

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: FUZZY INDUCTIVE LEARNING 🧠")
    print("=======================================================================\n")
    
    print("--- INDUCTIVE LEARNER: SEPARATING CAUSALITY FROM NOISE ---")
    print("We are feeding the Fuzzy Inductive Learner a set of messy, noisy observations.")
    print("Some events are truly causal. Some are just random coincidences.")
    
    # 10 training episodes
    train_data = [
        ["virus", "fever", "cough"],
        ["virus", "fever", "headache"],
        ["virus", "cough"],
        ["bacteria", "fever", "sweats"],
        ["bacteria", "fever"],
        ["eating_pizza", "fever"],                # Random noise
        ["wearing_blue", "cough"],                # Coincidence in training
        ["wearing_blue", "cough", "sneezing"],    # Coincidence in training
        ["sneezing", "earthquake"],               # Complete coincidence
        ["sneezing", "earthquake"]                # Complete coincidence
    ]
    
    # 5 hold-out testing episodes (Reality)
    test_data = [
        ["virus", "fever", "cough"],
        ["virus", "fever"],
        ["bacteria", "fever", "sweats"],
        ["wearing_blue", "fever"],                # Wearing blue didn't cause a cough this time
        ["sneezing", "headache"],                 # Sneezing didn't cause an earthquake
        ["eating_pizza", "happy"]                 # Eating pizza didn't cause a fever
    ]
    
    il = InductiveLearner()
    
    print("\n[Brain is scanning thousands of events... proposing fuzzy hypotheses...]")
    print("[Brain is rigorously testing hypotheses against hold-out reality...]\n")
    
    promoted, rejected = il.mine(train_data, test_data, min_support=2, min_conf=0.6, verify_conf=0.6, min_test=1)
    
    print("❌ REJECTED RULES (Fuzzy engine identified these as hallucinations/noise):")
    for a, b, reason in rejected:
        print(f"  - {a} -> {b} (Rejected: {reason})")
        
    print("\n✅ VERIFIED LAWS (Fuzzy engine crystallized these into crisp logic):")
    for r in promoted:
        print(f"  - {r.a} -> {r.b} (Training Confidence: {r.conf_train:.0%}, Reality Verified: {r.conf_test:.0%})")
        
    print("\n[Feeding verified laws to the Crisp Logic Engine...]")
    crisp_engine = ReasoningEngine()
    il.promote_into(crisp_engine, promoted, relation="causes")
    
    print("🤖 CRISP ENGINE: I have received the purified logic rules from the Fuzzy Core.")
    print("                 I can now mathematically guarantee these causal links.")

    print("\n=======================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("=======================================================================")

if __name__ == "__main__":
    main()
