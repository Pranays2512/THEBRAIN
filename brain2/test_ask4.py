import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from faculties.whole_brain import WholeBrain

b = WholeBrain()
queries = [
    "What causes pain?",
    "Which acid causes pain?",
    "What does acid cause?"
]
for q in queries:
    print(f"Q: {q}\nA:", b.ask(q), "\n")
