from faculties.whole_brain import WholeBrain

brain = WholeBrain()
q = "If one were to inquire about the geographical whereabouts of the institution known as ncert, what would be the most precise city of its location?"
ent = brain.front._entity(q)
srel, conf = brain.front.student.confident_rel(q)
print("Entity:", ent)
print("Student relation:", srel)
print("Student confidence:", conf)
print("Solver output:", brain.front._solve(ent, srel))
