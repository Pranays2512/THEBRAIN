#!/usr/bin/env python3
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from adapters.brain_interface import BrainInterface
from adapters.llm_adapter import StubClient, BrainQLMouth, BrainQLEyes

class MouthStubClient:
    def complete(self, prompt, system=""):
        if "TRUE" in prompt or "heartbeat" in prompt:
            return "Yes, Fluffy has a heartbeat. This is because Fluffy is a cat, cats are animals, and all animals have a heartbeat."
        if "red" in prompt:
            return "HCL turns litmus paper red. This is because HCL is an acid, and all acids turn litmus paper red."
        return "I don't know."

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN: LLM (Eyes/Mouth) + BRAIN (Logic) PIPELINE DEMO 🧠")
    print("=======================================================================\n")
    
    print("[1] Initializing pipeline...")
    
    # EYES LLM Mock: Translates natural language to BrainQL
    eyes_client = StubClient({
        "fluffy": "INHERIT Fluffy has",
        "hcl": "INHERIT HCL turns_litmus"
    })
    
    brain = BrainInterface(client=eyes_client)
    
    # MOUTH LLM Mock: Translates BrainQL Result JSON back into English
    brain._mouth = BrainQLMouth(client=MouthStubClient())
    
    # Load WholeBrain so logic works
    if not brain._wb:
        from faculties.whole_brain import WholeBrain
        brain._wb = WholeBrain(eyes=brain._eyes)
        brain._wb.kre.set_transitive("isa")
    
    print("\n[2] Teaching the Brain some basic facts (via logic nodes)...")
    
    # Teach basic facts directly to the graph
    brain.teach("cat", "isa", "animal")
    brain.teach("animal", "has", "heartbeat")
    brain.teach("Fluffy", "isa", "cat")
    
    brain.teach("acid", "turns_litmus", "red")
    brain.teach("HCL", "isa", "acid")
    
    dialogue = [
        "Does Fluffy have a heartbeat?",
        "What does HCL do to litmus paper?"
    ]
    
    print("\n[3] Conversing normally with the user...\n")
    for q in dialogue:
        print(f"👤 USER: \"{q}\"")
        
        # The pipeline: Text -> EYES (LLM parses to BrainQL) -> Logic Engine -> MOUTH (LLM verbalizes result)
        response = brain.respond(q)
        
        print(f"🤖 BRAIN: \"{response['reply']}\"")
        print(f"   (Internal Logic Engine Status: Verified = {response['verified']})\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
