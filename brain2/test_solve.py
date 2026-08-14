from faculties.whole_brain import WholeBrain

brain = WholeBrain()
q = "If one were to inquire about the geographical whereabouts of the institution known as ncert, what would be the most precise city of its location?"
print("ask_rich:", brain.ask_rich(q))
