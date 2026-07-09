# brain2/_refactor/baseline.py
"""Green + determinism screen. Run each candidate suite TWICE; keep only those that
exit 0 and produce identical output across runs (after normalizers). Write
baseline/<script>.out and baseline/manifest.json."""
import json, os, re, subprocess
BRAIN2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(BRAIN2, "baseline")
PY = os.path.join(BRAIN2, "..", "venv2", "bin", "python3")
CANDIDATES = ["harden_regress.py","exam.py","test_open_lang.py","test_phase_a.py",
              "stress_exam.py","reasoning_suite.py","component_validation.py","validate.py"]
NORMALIZERS = [r"\d+\.\d+s", r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", r"0x[0-9a-fA-F]+"]

def run(s):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
    r = subprocess.run([PY, s], cwd=BRAIN2, capture_output=True, text=True, env=env)
    out = r.stdout
    for pat in NORMALIZERS: out = re.sub(pat, "", out)
    return r.returncode, out

def main():
    os.makedirs(BASE, exist_ok=True)
    kept = []
    for s in CANDIDATES:
        if not os.path.exists(os.path.join(BRAIN2, s)):
            print(f"skip  {s} (absent)"); continue
        rc1, o1 = run(s); rc2, o2 = run(s)
        if rc1 == 0 and rc2 == 0 and o1 == o2:
            open(os.path.join(BASE, s + ".out"), "w").write(o1)
            kept.append({"script": s, "out": s + ".out", "normalizers": NORMALIZERS})
            print(f"keep  {s}")
        else:
            print(f"DROP  {s} (rc={rc1},{rc2} stable={o1==o2})")
    json.dump({"suites": kept}, open(os.path.join(BASE, "manifest.json"), "w"), indent=1)
    print(f"baselined {len(kept)} suites")

if __name__ == "__main__":
    main()
