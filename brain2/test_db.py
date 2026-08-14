from faculties.whole_brain import WholeBrain

brain = WholeBrain()
print("zinc in entities:", "zinc" in brain.entities)
print("litmus in entities:", "litmus" in brain.entities)
print("fastest in graph?", "fastest" in brain.entities or "fastest" in brain.relations)
