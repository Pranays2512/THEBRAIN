import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from faculties.whole_brain import WholeBrain

b = WholeBrain()
print("planner:", b.planner.try_answer("what is a family?"))
print("lang:", b.lang.respond("what is a family?"))
