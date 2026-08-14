#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from faculties.appraisal_engine import AppraisalEngine
from faculties.autonomous_loop import run as autonomous_run

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: EMOTIONS, CURIOSITY, & DAYDREAMING 🧠")
    print("=======================================================================\n")
    
    print("--- PART 1: 'EMOTION' (Pragmatic Appraisal Engine) ---")
    print("In THEBRAIN, emotion isn't a 'feeling'—it's a high-speed pragmatic appraisal.")
    print("It grades text on dimensions (greeting, command, question) before reading it,\nallowing the Brain to understand INTENT before it processes MEANING.\n")
    
    appraiser = AppraisalEngine()
    
    inputs = [
        "Hey! How are you?",
        "Please explain gravity to me.",
        "The sky is blue."
    ]
    
    for text in inputs:
        print(f"👤 USER: \"{text}\"")
        appraisal = appraiser.appraise(text)
        # Manually format the appraisal output to look clean
        active = {k: round(v, 2) for k, v in appraisal.frame.items() if v > 0}
        print(f"🧠 APPRAISAL: Detected Type -> [{appraisal.type.upper()}]")
        print(f"   Active Dimensions: {active}\n")

    print("\n--- PART 2: 'DAYDREAMING & CURIOSITY' (The Autonomous Loop) ---")
    print("When you stop talking to the Brain, it doesn't just turn off. It 'daydreams'.")
    print("Curiosity: It finds a gap in its knowledge.")
    print("Daydream: It generates mathematical hypotheses and tests them in its sandbox.")
    print("Learning: If a hypothesis survives, it stores it and learns its shape to guess faster next time.\n")
    
    # Running the autonomous loop module directly
    autonomous_run()
    
    print("\n=======================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("=======================================================================")

if __name__ == "__main__":
    main()
