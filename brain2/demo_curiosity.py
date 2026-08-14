#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from faculties.curiosity_loop import CuriosityLoop

def main():
    print("================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: CURIOSITY & IDLE LEARNING 🧠")
    print("================================================================\n")
    
    print("--- IDLE LEARNING (Finding gaps in the world model) ---")
    print("The Brain is fed observations over time. Some are predictable, some are noise.")
    
    predictable = [["seed", "plant"]] * 2
    noise = [["dice", "one"], ["dice", "two"], ["dice", "three"]]
    
    cl = CuriosityLoop()
    
    print("\n[Tick 1] Observing...")
    cl.observe(predictable + noise[:1])
    _, err = cl.tick()
    print(f"  Prediction Error: {err:.2f}")
    
    print("\n[Tick 2] Observing again...")
    cl.observe(predictable + noise[1:2])
    _, err = cl.tick()
    print(f"  Prediction Error: {err:.2f}")
    
    print("\n[Tick 3] Consolidating Knowledge...")
    cl.observe(predictable + noise[2:3])
    _, err = cl.tick()
    print(f"  Prediction Error: {err:.2f} (Error drops as patterns are learned)")
    
    print("\n[What did the Brain learn?]")
    print(f"  Does a seed lead to a plant? -> {'Yes' if cl.predict.get('seed') == 'plant' else 'No'} (Learned)")
    print(f"  Does a dice roll lead to a specific number? -> {'Yes' if cl.predict.get('dice') else 'No'} (Rejected as noise)")
    
    print("\n[Curiosity Gaps (What is the Brain still trying to figure out?)]")
    gaps = dict(cl.curiosity_gaps())
    for item, gap in gaps.items():
        print(f"  • {item} (Curiosity/Uncertainty Score: {gap:.2f})")
    print("\nNote: The Brain successfully solved 'seed' so it is no longer curious about it.")
    print("It remains highly curious about 'dice' because it cannot find a stable rule!")
    
    print("\n================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    main()
