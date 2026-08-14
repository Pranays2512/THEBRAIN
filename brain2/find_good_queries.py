from faculties.whole_brain import WholeBrain
import random

brain = WholeBrain()

print("--- Known Relations ---")
print(list(brain.front.solvable)[:10])

print("\n--- Example Facts ---")
count = 0
for s, r, o in brain.fkb.kb.facts:
    if r not in ("isa", "same_as") and len(s) > 3 and len(o) > 3:
        print(f"Fact: {s} -> {r} -> {o}")
        count += 1
        if count >= 20:
            break

print("\n--- Objects with multiple relations (Good for Query Planner) ---")
count = 0
for s in list(brain.entities)[:500]:
    rels = brain.fkb.relations_into(s)
    if len(rels) > 0:
        for r in set(rels):
            subjects = brain.fkb.subjects_with(r, s)
            if len(subjects) > 2:
                print(f"Query: In how many ways can we get {s.replace('_', ' ')}? (Relation: {r}) -> {subjects[:3]}")
                count += 1
                break
    if count >= 5:
        break
