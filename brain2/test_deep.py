from faculties.whole_brain import WholeBrain
from engines.reasoning import deeper_grammar as DG

brain = WholeBrain()
DG.ENTS = set(brain.entities)
DG.RELS = set(brain.relations)

q = "If a substance turns red litmus blue, what is its nature?"
print("DeeperParser:", DG.DeeperParser(brain.fkb, brain.mem).answer(q))
