# brain2/_refactor/gate.py
"""Run every baselined suite; diff stdout against baseline/. Exit 1 on any diff."""
import json, os, subprocess, sys
BRAIN2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(BRAIN2, "baseline")
PY = os.path.join(BRAIN2, "..", "venv2", "bin", "python3")

def run(script, norm):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1",
               PYTHONHASHSEED="0")
    r = subprocess.run([PY, script], cwd=BRAIN2, capture_output=True, text=True, env=env)
    out = r.stdout
    for pat in norm:                       # strip volatile lines
        import re; out = re.sub(pat, "", out)
    return out

def main():
    man = json.load(open(os.path.join(BASE, "manifest.json")))
    bad = 0
    for entry in man["suites"]:
        cur = run(entry["script"], entry.get("normalizers", []))
        want = open(os.path.join(BASE, entry["out"])).read()
        if cur != want:
            bad += 1
            print(f"DIFF  {entry['script']}")
            import difflib
            for line in list(difflib.unified_diff(want.splitlines(), cur.splitlines(), lineterm=""))[:8]:
                print("   " + line)
        else:
            print(f"ok    {entry['script']}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
