#!/usr/bin/env python3
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from adapters.brain_interface import BrainInterface
from adapters.llm_adapter import OllamaClient, SafeClient

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN: LLM (Eyes/Mouth) + BRAIN (Logic) PIPELINE DEMO 🧠")
    print("=======================================================================\n")
    
    print("[1] Initializing pipeline...")
    # SafeClient falls back to a deterministic path if the local Ollama server is offline
    client = SafeClient(OllamaClient("qwen3:1.7B"))
    brain = BrainInterface(client=client)
    
    print("\n[2] Teaching the Brain some basic facts directly...")
    
    # Teach basic facts
    brain.teach("cat", "isa", "animal")
    brain.teach("animal", "has", "heartbeat")
    brain.teach("Fluffy", "isa", "cat")
    
    brain.teach("acid", "turns_litmus", "red")
    brain.teach("HCL", "isa", "acid")
    
    dialogue = [
        "Does Fluffy have a heartbeat?",
        "What does HCL do to litmus paper?",
        "Can you combine math and physics to solve an equation?"
    ]
    
    print("\n[3] Conversing normally with the user...\n")
    for q in dialogue:
        print(f"👤 USER: {q}")
        # The pipeline: Text -> EYES (LLM parses to BrainQL) -> Logic Engine -> MOUTH (LLM verbalizes result)
        response = brain.respond(q)
        print(f"🤖 BRAIN: {response}\n")

if __name__ == "__main__":
    main()
