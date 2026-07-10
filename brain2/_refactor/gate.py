# brain2/_refactor/gate.py
"""Run every baselined suite; diff stdout against baseline/. Exit 1 on any diff."""
import json, os, subprocess, sys
BRAIN2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(BRAIN2, "baseline")
PY = os.path.join(BRAIN2, "..", "venv2", "bin", "python3")

def resolve(script):
    """Run bare 'exam.py' if at root; else find it in a package and run '-m pkg.mod'
    (keeps brain2/ on sys.path, unlike `python3 tests/exam.py`)."""
    if os.path.exists(os.path.join(BRAIN2, script)):
        return [PY, script]
    for dp, _, fs in os.walk(BRAIN2):
        if "/_refactor" in dp or "/.git" in dp or "/__pycache__" in dp:
            continue
        if script in fs:
            rel = os.path.relpath(os.path.join(dp, script), BRAIN2)
            return [PY, "-m", rel[:-3].replace(os.sep, ".")]
    return [PY, script]

def run(script, norm):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1",
               PYTHONHASHSEED="0")
    r = subprocess.run(resolve(script), cwd=BRAIN2, capture_output=True, text=True, env=env)
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
