"""
Run this from your THEBRAIN folder to confirm you have the right files.
Usage: python CHECK_FILES.py
"""
import os, re

checks = {
    "brain_in_world3.py": [
        ("l4_position import",    r"from l4_position import"),
        ("epsilon_override set",  r"brain\.action\.epsilon_override"),
        ("e_food_found flag",     r"e_food_found"),
        ("K wall surgery",        r"world\.current_node == .K. and info\[.wall_hit.\]"),
        ("E first-find replay",   r"reward \* 5\.0"),
    ],
    "m56_action.py": [
        ("epsilon_override field", r"self\.epsilon_override = -1\.0"),
        ("epsilon warmup",         r"EPSILON_WARMUP_STEPS"),
        ("3b-wall direct penalty", r"3b-wall"),
        ("Q_n no rpe guard",       r"Updated on all RPE"),
        ("replay alpha 0.20",      r"REPLAY_ALPHA\s*=\s*0\.20"),
    ],
    "brain.py": [
        ("PositionBelief import",  r"from l4_position import PositionBelief"),
    ],
    "m57_planner.py": [
        ("warmup 80k",             r"M57_WARMUP_STEPS\s*=\s*80_000"),
    ],
}

all_ok = True
for fname, tests in checks.items():
    if not os.path.exists(fname):
        print(f"❌ {fname} NOT FOUND in current directory")
        all_ok = False
        continue
    content = open(fname).read()
    file_ok = True
    for label, pattern in tests:
        found = bool(re.search(pattern, content))
        if not found:
            print(f"❌ {fname}: MISSING — {label}")
            all_ok = False
            file_ok = False
    if file_ok:
        print(f"✓  {fname}: all {len(tests)} checks passed")

print()
if all_ok:
    print("✅ ALL FILES CORRECT — run python brain_in_world3.py")
else:
    print("⚠️  REPLACE the failing files with the downloaded outputs before running.")