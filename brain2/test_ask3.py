import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from faculties.whole_brain import WholeBrain

b = WholeBrain()
queries = [
    "Is a dog a living thing?",
    "What is a dog?",
    "What is the force of a rocket?",
    "Does acid cause pain?",
    "What is the ph of pure water?"
]
for q in queries:
    print(f"Q: {q}\nA:", b.ask(q), "\n")
