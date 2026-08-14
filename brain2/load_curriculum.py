import json
import os

path = "data/brain_curriculum.txt"
facts = {}
isas = {}

with open(path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("ISA: "):
            parts = line[5:].split(" | ")
            if len(parts) == 2:
                # Add to isas (simple dict of child -> parent)
                isas[parts[0]] = parts[1]
        elif line.startswith("FACT: "):
            parts = line[6:].split(" | ")
            if len(parts) == 3:
                ent_rel = f"{parts[0]}|{parts[1]}"
                facts[ent_rel] = parts[2]

store_dir = "engines/store/brain_store"
os.makedirs(store_dir, exist_ok=True)

with open(os.path.join(store_dir, "facts.json"), "w") as f:
    json.dump(facts, f, indent=2)
    
print(f"Loaded {len(facts)} facts and {len(isas)} ISAs into BrainStore.")
