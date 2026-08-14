from faculties.query_planner import QueryPlanner
from faculties.whole_brain import WholeBrain

brain = WholeBrain()
qp = brain.planner
q = "If one were to inquire about the geographical whereabouts of the institution known as ncert, what would be the most precise city of its location?"
print("queries:", qp.parse(q))
