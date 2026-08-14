#!/usr/bin/env python3
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from adapters.brain_interface import BrainInterface
from engines.reasoning.brainql import BrainQLExecutor
from adapters.llm_adapter import StubClient, BrainQLEyes, BrainQLMouth

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN: EYES -> BRAIN -> MOUTH PIPELINE DEMO 🧠")
    print("=======================================================================\n")
    
    mock_llm_responses = {
        # EYES mocking (User -> BrainQL)
        "fluffy": "```brainql\nINHERIT Fluffy has\n```",
        "hcl": "```brainql\nINHERIT HCL turns_litmus\n```",
        
        # MOUTH mocking (BrainQLResult -> Fluent English)
        "Fluffy": "Yes, Fluffy has a heartbeat. This is because Fluffy is a cat, cats are animals, and all animals have a heartbeat.",
        "HCL": "HCL turns litmus paper red. This is because HCL is an acid, and all acids turn litmus paper red."
    }
    
    client = StubClient(mock_llm_responses)
    brain = BrainInterface(client=client)
    
    if not brain._wb:
        from faculties.whole_brain import WholeBrain
        brain._wb = WholeBrain(eyes=brain._eyes)
        brain._wb.kre.set_transitive("isa")
    
    brain.teach("cat", "isa", "animal")
    brain.teach("animal", "has", "heartbeat")
    brain.teach("Fluffy", "isa", "cat")
    
    brain.teach("acid", "turns_litmus", "red")
    brain.teach("HCL", "isa", "acid")
    
    print("Facts taught to Brain graph.")
    
    dialogue = [
        "Does Fluffy have a heartbeat?",
        "What does HCL do to litmus paper?"
    ]
    
    for q in dialogue:
        print(f"\n👤 USER: {q}")
        print("👀 EYES (LLM): Translating messy text into structured BrainQL...")
        
        bql = brain._eyes.parse(q)
        print(f"   -> Generated: {bql}")
        
        print("🧠 BRAIN (Logic): Executing BrainQL graph query...")
        if isinstance(bql, list):
            res = brain._wb.execute_bql(bql)
        else:
            res = brain._wb.execute_bql([bql])
            
        print(f"   -> Result from graph: {res}")
        
        print("👄 MOUTH (LLM): Translating structured result back into fluent English...")
        out = brain._mouth.render(res)
        print(f"🤖 RESPONSE: {out}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
