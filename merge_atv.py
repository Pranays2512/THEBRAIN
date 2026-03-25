import re

files = ["attention.py", "thought.py", "valence.py"]
imports = ["import numpy as np", "import math", "from collections import deque"]

content = "\n".join(imports) + "\n\n"

for fname in files:
    with open(fname, "r") as f:
        text = f.read()
    
    # Remove imports
    text = re.sub(r'^import .*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^from .*$', '', text, flags=re.MULTILINE)
    
    # Optional: deduplicate _build_grid_dist_sq
    # We can just leave it or rename, but if it is identical, we can just replace the second occurrence.
    
    content += f"\n# {'='*60}\n# FROM {fname}\n# {'='*60}\n" + text.strip() + "\n"

# Remove duplicate _GRID_DIST_SQ and _build_grid_dist_sq definitions
# To be safe and simple, let's just let them redefine it, it's identical logic and fast.

with open("evaluators.py", "w") as f:
    f.write(content)
