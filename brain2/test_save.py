import os
import sys

# Add brain2 directory to sys.path so we can import brain2 and train.py functions
sys.path.append('/Users/pranay./Documents/THEBRAIN/brain2')

import brain2
from train.train import save_checkpoint

print("Initializing Brain...")
cfg = {
    'som_rows': 64,
    'som_cols': 64,
    'n_dims': 64,
    'hidden_dim': 512
}
b = brain2.Brain(**cfg)

print("Testing save_checkpoint...")
try:
    save_checkpoint(b, 'test_checkpoint', 'test')
    print("SUCCESS: save_checkpoint executed without crashing!")
except Exception as e:
    print(f"FAILED: {e}")
