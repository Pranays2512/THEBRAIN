import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from faculties.whole_brain import WholeBrain

b = WholeBrain()
print("Ask:", b.ask("what is a family?"))
print("Entities:", "family" in b.entities)
print("Concepts:", "family" in b.concepts)
