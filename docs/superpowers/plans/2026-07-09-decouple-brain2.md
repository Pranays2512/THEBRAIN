# Decouple brain2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `brain2/`'s 132 flat Python files into a layered package tree (core → io → faculties → training → tests, plus experimental/cpp) with zero change to logic, execution flow, or runtime behavior.

**Architecture:** Moves + import-statement rewrites + pure re-export facades only. The layering is a true partition of the actual acyclic import DAG. A codemod performs every import rewrite deterministically; a full output-diff gate runs after every package move and must produce byte-identical output against a pre-refactor baseline. Root `runpy` shims keep every existing run command and the settings.json allow-list working.

**Tech Stack:** Python 3, libcst (import codemod), git, the existing brain2 acceptance suites.

**Spec:** `docs/superpowers/specs/2026-07-09-decouple-brain2-design.md`

**Working directory for all commands:** `brain2/` (cwd on `sys.path`). Interpreter: `../venv2/bin/python3`.

> **Isolation note:** Background training jobs may still be writing runtime state (`brain_store/`, `trained/`). Before capturing baselines (Task 3), confirm no training process is running (`ps aux | grep -E "read_pdf_train|train_"`). The refactor only moves committed source; it does not touch runtime artifacts. Running the refactor in a git worktree (superpowers:using-git-worktrees) is recommended so it stays isolated from live jobs — optional.

---

## File Structure

Tooling (temporary, lives under `brain2/_refactor/`, removed in the final task):
- `_refactor/classify.py` — assigns every module to a layer/subpackage; writes `move_map.json`; prints the grouping + up-edges for human review.
- `_refactor/codemod.py` — physically relocates a batch's files (`git mv`), creates `__init__.py` facades, and rewrites all import sites (libcst) to point at moved modules' new package paths. Idempotent.
- `_refactor/gate.py` — runs each baselined suite and diffs stdout against `baseline/`; exits nonzero on any diff.
- `_refactor/baseline.py` — green+determinism screen; captures `baseline/*.out` and `baseline/manifest.json`.

Target tree (produced incrementally):
```
brain2/
  core/{reasoning,synthesis,math,knowledge,grounding,events,neural,store}/
  io/  faculties/  training/  tests/  experimental/  cpp/
  <root shims>.py           # runpy entrypoints for directly-invoked scripts
  brain2*.so                # native module — stays put, `import brain2` unchanged
```

---

## Task 1: Classifier — produce and review the move map

**Files:**
- Create: `brain2/_refactor/classify.py`
- Create (output): `brain2/_refactor/move_map.json`

- [ ] **Step 1: Write the classifier**

```python
# brain2/_refactor/classify.py
"""Assign every top-level brain2 module to a layer/subpackage from real imports +
name heuristics. Writes move_map.json {old_module: 'pkg/sub'} and prints groups +
up-pointing edges for human review. Pure analysis — moves nothing."""
import ast, json, os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # brain2/

def modules():
    out = {}
    for f in os.listdir(ROOT):
        if f.endswith(".py") and f != "__init__.py":
            p = os.path.join(ROOT, f)
            try: tree = ast.parse(open(p, errors="ignore").read())
            except Exception: continue
            deps = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for a in n.names: deps.add(a.name.split(".")[0])
                elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    deps.add(n.module.split(".")[0])
            out[f[:-3]] = deps
    return out

TEST = re.compile(r"^(test_|harden_|stress_|.*_exam$)|^(validate|component_validation|reasoning_suite|exam|exam_math)$")
TRAIN = re.compile(r"^(train_|read_pdf_train$|auto_|knowledge_distill$|student_)")
IO = {"llm_adapter","llm_extractor","mouth","ocr_pdf","nl_front","server","chat",
      "brain_repl","structural_parser","math_parser","integrated_front","converse"}
FAC = {"whole_brain","read_book","reading_loop","conversation_engine","query_planner",
       "neuro_bridge","event_predict","feature_learner"}
# core subdomain keyword buckets (first match wins, order matters)
CORE_SUB = [
 ("synthesis", re.compile(r"synth|proposer|program_synth|loop_synth|dp_|composable|refut|invariant|conjecture|code_gen|codegen")),
 ("math", re.compile(r"algebra|calculus|integral|physics|word_math|factoriz|dimensional|prob_compute|math_")),
 ("grounding", re.compile(r"ground|grounding|crispify|context_embed|domain_features")),
 ("events", re.compile(r"event_|verb_|discourse|analogy|compositional")),
 ("knowledge", re.compile(r"knowledge|concept|semantic|world_knowledge|core_knowledge|fact_extractor|conceptnet|taxonom")),
 ("neural", re.compile(r"neural_lm|cpp_accel")),
 ("store", re.compile(r"brain_store|check_library|template_memory|type_oracle|parse_template|corpus_scale|coverage_harness")),
 ("reasoning", re.compile(r"reason|tree_|means_ends|nested_parser|deeper_grammar|dual_process|planning|planner|learned_guidance")),
]

def layer_of(m):
    if TEST.match(m): return "tests"
    if TRAIN.match(m): return "training"
    if m in IO: return "io"
    if m in FAC: return "faculties"
    for sub, rx in CORE_SUB:
        if rx.search(m): return f"core/{sub}"
    return "core/misc"

def main():
    mods = modules()
    mp = {m: layer_of(m) for m in mods}
    groups = collections.defaultdict(list)
    for m, dst in mp.items(): groups[dst].append(m)
    for dst in sorted(groups):
        print(f"\n{dst} ({len(groups[dst])}): " + ", ".join(sorted(groups[dst])))
    # up-pointing edges (rank: core<io<faculties<training<tests)
    rank = lambda d: {"io":1,"faculties":2,"training":3,"tests":4}.get(d.split("/")[0], 0)
    local = set(mods); ups = []
    for m, ds in mods.items():
        for d in ds & local - {m}:
            if rank(mp[d]) > rank(mp[m]): ups.append((m, mp[m], d, mp[d]))
    print(f"\nUP-EDGES to review ({len(ups)}):")
    for m, lm, d, ld in ups: print(f"  {m}({lm}) -> {d}({ld})")
    json.dump(mp, open(os.path.join(os.path.dirname(__file__), "move_map.json"), "w"), indent=1)
    print(f"\nwrote move_map.json ({len(mp)} modules)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd brain2 && ../venv2/bin/python3 _refactor/classify.py`
Expected: prints each package group, the up-edge list, and `wrote move_map.json (132 modules)`.

- [ ] **Step 3: Human review of the map (GATE — do not proceed until resolved)**

Read the printed groups and `move_map.json`. For every module, confirm the destination is right. Pay special attention to:
- `core/misc` entries — each must be reassigned to a real subpackage or justified.
- Every up-edge — decide per edge: relocate the module to the layer its edges point to (edit its entry in `move_map.json`), or mark it a documented exception (add to `_refactor/exceptions.md` with the reason).
Hand-edit `move_map.json` until the map is correct and `core/misc` is empty.

- [ ] **Step 4: Commit the reviewed map**

```bash
cd brain2
git add _refactor/classify.py _refactor/move_map.json _refactor/exceptions.md
git commit -m "refactor(brain2): reviewed module -> package map"
```

---

## Task 2: Build the codemod, gate, and baseline tools

**Files:**
- Create: `brain2/_refactor/codemod.py`, `brain2/_refactor/gate.py`, `brain2/_refactor/baseline.py`

- [ ] **Step 1: Install libcst into venv2**

Run: `../venv2/bin/python3 -m pip install --quiet libcst && ../venv2/bin/python3 -c "import libcst; print('libcst', libcst.__version__)"`
Expected: prints a version.

- [ ] **Step 2: Write the codemod**

```python
# brain2/_refactor/codemod.py
"""Relocate the modules for the given destination prefixes into their package dirs,
create __init__.py facades, then rewrite EVERY import of ANY already-moved module to
its new package path. Idempotent: re-running after more moves fixes all imports.

    ../venv2/bin/python3 _refactor/codemod.py core/store core/neural   # activate batches
"""
import json, os, subprocess, sys
import libcst as cst

BRAIN2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = json.load(open(os.path.join(os.path.dirname(__file__), "move_map.json")))

def dotted(dst):            # 'core/store' -> 'core.store'
    return dst.replace("/", ".")

def move_files(active_prefixes):
    """git mv modules whose dst is under an active prefix and are still at root."""
    for mod, dst in MAP.items():
        if not any(dst == p or dst.startswith(p + "/") or p == dst.split("/")[0] for p in active_prefixes):
            continue
        src = os.path.join(BRAIN2, mod + ".py")
        if not os.path.exists(src):
            continue                       # already moved
        pkgdir = os.path.join(BRAIN2, dst)
        os.makedirs(pkgdir, exist_ok=True)
        # ensure __init__.py chain exists
        parts = dst.split("/"); cur = BRAIN2
        for part in parts:
            cur = os.path.join(cur, part)
            init = os.path.join(cur, "__init__.py")
            if not os.path.exists(init):
                open(init, "w").close()
        subprocess.check_call(["git", "mv", src, os.path.join(pkgdir, mod + ".py")], cwd=BRAIN2)

def moved_now():
    """{module: dotted_pkg} for every module physically living in a package dir."""
    out = {}
    for mod, dst in MAP.items():
        if os.path.exists(os.path.join(BRAIN2, dst, mod + ".py")):
            out[mod] = dotted(dst)
    return out

class Rewriter(cst.CSTTransformer):
    def __init__(self, moved):  # moved: {module: dotted_pkg}
        self.moved = moved
    def leave_Import(self, orig, updated):
        new = []
        for a in updated.names:
            name = a.name.value if isinstance(a.name, cst.Name) else None
            if name in self.moved:
                # import X [as Z]  ->  from pkg import X [as Z]
                imp = cst.ImportFrom(
                    module=cst.parse_expression(self.moved[name]),
                    names=[cst.ImportAlias(name=cst.Name(name), asname=a.asname)])
                new.append(("from", imp))
            else:
                new.append(("plain", a))
        if all(k == "plain" for k, _ in new):
            return updated
        # split into separate statements; libcst handles one node -> use FlattenSentinel
        stmts = []
        plains = [v for k, v in new if k == "plain"]
        if plains:
            stmts.append(cst.Import(names=[cst.ImportAlias(name=p.name, asname=p.asname) for p in plains]))
        stmts += [v for k, v in new if k == "from"]
        return cst.FlattenSentinel(stmts)
    def leave_ImportFrom(self, orig, updated):
        if updated.module and isinstance(updated.module, cst.Name):
            m = updated.module.value
            if m in self.moved:  # from X import a  ->  from pkg.X import a
                return updated.with_changes(module=cst.parse_expression(self.moved[m] + "." + m))
        return updated

def rewrite_all(moved):
    for dirpath, _, files in os.walk(BRAIN2):
        if "/.git" in dirpath or "/_refactor" in dirpath:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            src = open(p, encoding="utf-8").read()
            try:
                tree = cst.parse_module(src)
                out = tree.visit(Rewriter(moved)).code
            except Exception as e:
                print(f"  SKIP {p}: {e}"); continue
            if out != src:
                open(p, "w", encoding="utf-8").write(out)

if __name__ == "__main__":
    active = sys.argv[1:]
    if active:
        move_files(active)
    rewrite_all(moved_now())
    print("codemod done; active:", active or "(rewrite-only)")
```

- [ ] **Step 3: Write the gate**

```python
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
            # first differing line for a quick signal
            import difflib
            for line in list(difflib.unified_diff(want.splitlines(), cur.splitlines(), lineterm=""))[:8]:
                print("   " + line)
        else:
            print(f"ok    {entry['script']}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the baseline capture**

```python
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
```

- [ ] **Step 5: Commit the tooling**

```bash
cd brain2
git add _refactor/codemod.py _refactor/gate.py _refactor/baseline.py
git commit -m "refactor(brain2): codemod + gate + baseline tooling"
```

---

## Task 3: Capture the baseline (green + determinism screen)

**Files:**
- Create (output): `brain2/baseline/*.out`, `brain2/baseline/manifest.json`

- [ ] **Step 1: Confirm no training jobs are writing state**

Run: `ps aux | grep -E "[r]ead_pdf_train|[t]rain_" ; echo done`
Expected: no matching process lines before `done`. If any run, wait for them or stop them before continuing (baseline must be on a stable tree).

- [ ] **Step 2: Capture baselines**

Run: `cd brain2 && ../venv2/bin/python3 _refactor/baseline.py`
Expected: a `keep`/`DROP` line per candidate and `baselined N suites` with N ≥ 1. Note any DROP (suite excluded from the gate — that's fine, recorded in the manifest).

- [ ] **Step 3: Prove the gate is green on the untouched tree**

Run: `cd brain2 && ../venv2/bin/python3 _refactor/gate.py; echo "EXIT=$?"`
Expected: `ok` for every kept suite and `EXIT=0`. If not green here, the baseline is broken — fix before any move.

- [ ] **Step 4: Commit the baseline**

```bash
cd brain2
git add baseline/
git commit -m "refactor(brain2): captured acceptance baseline"
```

---

## Tasks 4–16: Move packages one at a time (bottom-up)

Each task moves one batch, runs the codemod, and requires a byte-identical gate before committing. The batch order respects the dependency direction so imports only ever rewrite downward.

**Canonical per-package procedure (used by every task below):**

```bash
cd brain2
../venv2/bin/python3 _refactor/codemod.py <PREFIX>      # git mv batch + create __init__ + rewrite imports
../venv2/bin/python3 _refactor/gate.py; echo "EXIT=$?"  # MUST print EXIT=0
```
- If `EXIT=0`: commit. `git add -A && git commit -m "refactor(brain2): move <PREFIX>"`
- If `EXIT≠0`: `git reset --hard HEAD` (revert the whole batch), read the DIFF output, fix the cause (usually a mis-mapped module or an import form the codemod skipped — check the `SKIP` lines), then retry. **Never commit a non-green move.**

- [ ] **Task 4 — `core/store`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/store`
  - Run gate; expect `EXIT=0`. Commit `refactor(brain2): move core/store`.

- [ ] **Task 5 — `core/neural`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/neural`
  - Run gate; expect `EXIT=0`. Commit `refactor(brain2): move core/neural`.
  - Note: `import brain2` (the `.so`) must be untouched — confirm it does not appear in any codemod rewrite (it's not in `move_map.json`).

- [ ] **Task 6 — `core/math`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/math` ; gate `EXIT=0` ; commit `move core/math`.

- [ ] **Task 7 — `core/events`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/events` ; gate `EXIT=0` ; commit `move core/events`.

- [ ] **Task 8 — `core/grounding`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/grounding` ; gate `EXIT=0` ; commit `move core/grounding`.

- [ ] **Task 9 — `core/knowledge`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/knowledge` ; gate `EXIT=0` ; commit `move core/knowledge`.

- [ ] **Task 10 — `core/reasoning`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/reasoning` ; gate `EXIT=0` ; commit `move core/reasoning`.

- [ ] **Task 11 — `core/synthesis`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py core/synthesis` ; gate `EXIT=0` ; commit `move core/synthesis`.

- [ ] **Task 12 — `io`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py io` ; gate `EXIT=0` ; commit `move io`.

- [ ] **Task 13 — `faculties`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py faculties` ; gate `EXIT=0` ; commit `move faculties`.

- [ ] **Task 14 — `training`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py training` ; gate `EXIT=0` ; commit `move training`.
  - Note: entrypoints moved here lose their root path; the gate still passes because it invokes the *test* suites, not the training scripts. Root shims are restored in Task 17.

- [ ] **Task 15 — `tests`**
  - Run: `../venv2/bin/python3 _refactor/codemod.py tests` ; gate `EXIT=0` ; commit `move tests`.
  - Note: `gate.py` and `baseline.py` reference suites by bare script name (`harden_regress.py`). After this move they live under `tests/`. Update `manifest.json` entries' `script` field to `tests/<name>` in the same commit (the gate must keep finding them). Re-run the gate after the edit; still `EXIT=0`.

- [ ] **Task 16 — `experimental`**
  - Before moving: run `../venv2/bin/python3 _refactor/classify.py` logic is not needed here; instead identify quarantine candidates: modules still at root with zero inbound edges from any moved package and not an entrypoint. Command:
    ```bash
    cd brain2 && ../venv2/bin/python3 - <<'PY'
    import ast, os
    root=[f[:-3] for f in os.listdir('.') if f.endswith('.py') and f!='__init__.py']
    inbound={m:0 for m in root}
    for dp,_,fs in os.walk('.'):
        if '/.git' in dp or '/_refactor' in dp: continue
        for f in fs:
            if not f.endswith('.py'): continue
            try: t=ast.parse(open(os.path.join(dp,f),errors='ignore').read())
            except Exception: continue
            for n in ast.walk(t):
                mods=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else \
                     ([n.module.split('.')[0]] if isinstance(n,ast.ImportFrom) and n.module and n.level==0 else [])
                for mm in mods:
                    if mm in inbound: inbound[mm]+=1
    print("QUARANTINE CANDIDATES (0 inbound, still at root):")
    for m,c in sorted(inbound.items()):
        if c==0: print(" ", m)
    PY
    ```
  - **Human review:** confirm the candidate list is genuinely one-off/dead (not a live entrypoint). Add each approved module to `move_map.json` with dst `experimental`, then run `../venv2/bin/python3 _refactor/codemod.py experimental`.
  - Run gate; expect `EXIT=0`. Commit `refactor(brain2): quarantine experimental`.

---

## Task 17: Root shims — restore identical invocation

**Files:**
- Create: one `brain2/<name>.py` shim per directly-invoked entrypoint now living under `training/`, `tests/`, or `io/`.

- [ ] **Step 1: Generate shims for every moved entrypoint**

```bash
cd brain2
../venv2/bin/python3 - <<'PY'
import json, os
MAP=json.load(open("_refactor/move_map.json"))
# entrypoints = modules whose file has an `if __name__ == "__main__"` guard
ENTRY=[]
for mod,dst in MAP.items():
    p=os.path.join(dst,mod+".py")
    if os.path.exists(p) and '__main__' in open(p,errors='ignore').read():
        ENTRY.append((mod,dst.replace("/",".")))
for mod,pkg in ENTRY:
    shim=f"{mod}.py"
    if os.path.exists(shim):  # never clobber a real root file
        continue
    open(shim,"w").write(
        "import runpy\n"
        f"runpy.run_module('{pkg}.{mod}', run_name='__main__')\n")
    print("shim", shim, "->", f"{pkg}.{mod}")
PY
```
Expected: a `shim` line for each entrypoint (`train_all.py -> training.train_all`, `harden_regress.py -> tests.harden_regress`, `read_pdf_train.py -> training.read_pdf_train`, `ocr_pdf.py -> io.ocr_pdf`, etc.).

- [ ] **Step 2: Verify a sample of allow-listed commands still run**

Run: `cd brain2 && ../venv2/bin/python3 harden_regress.py > /tmp/shim_check.out 2>&1; echo "EXIT=$?" && ../venv2/bin/python3 -c "print('shim import ok')"`
Expected: `EXIT=0` and identical behavior to the pre-refactor run (spot-check against `baseline/harden_regress.py.out` if that suite was kept).

- [ ] **Step 3: Full gate one more time (shims present)**

Run: `cd brain2 && ../venv2/bin/python3 _refactor/gate.py; echo "EXIT=$?"`
Expected: `EXIT=0`.

- [ ] **Step 4: Commit**

```bash
cd brain2 && git add -A && git commit -m "refactor(brain2): root runpy shims keep invocation identical"
```

---

## Task 18: Facade public APIs, architecture map, cleanup

**Files:**
- Modify: each package `__init__.py` (add re-exports)
- Create: `brain2/ARCHITECTURE.md`
- Remove: `brain2/_refactor/` (tooling), `brain2/baseline/` (optional keep)

- [ ] **Step 1: Populate facades with public re-exports**

For each package, edit its `__init__.py` to re-export the names that code OUTSIDE the package imports from its modules. This script prints the exact `from .<module> import <Name>` line to add, per package:

```bash
cd brain2 && ../venv2/bin/python3 - <<'PY'
import ast, os, collections
LAYERS = ("core", "io", "faculties", "training")
# pkg -> module -> set(names)  imported from OUTSIDE the pkg
want = collections.defaultdict(lambda: collections.defaultdict(set))
for dp, _, fs in os.walk('.'):
    if '/.git' in dp or '/_refactor' in dp:
        continue
    for f in fs:
        if not f.endswith('.py'):
            continue
        importer_pkg = "/".join(os.path.relpath(os.path.join(dp, f), '.').split(os.sep)[:-1])
        try:
            t = ast.parse(open(os.path.join(dp, f), errors='ignore').read())
        except Exception:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                parts = n.module.split('.')
                if parts[0] not in LAYERS or len(parts) < 2:
                    continue
                pkg, module = "/".join(parts[:-1]), parts[-1]   # e.g. core/store , brain_store
                if importer_pkg == pkg:                          # skip intra-package imports
                    continue
                for a in n.names:
                    want[pkg][module].add(a.name)
for pkg in sorted(want):
    print(f"\n# {pkg}/__init__.py")
    for module in sorted(want[pkg]):
        names = ", ".join(sorted(want[pkg][module]))
        print(f"from .{module} import {names}")
PY
```
Paste the printed `from .<module> import <Name>` lines into each package's `__init__.py`. Run the gate after editing (`EXIT=0`) — facades are additive re-exports, so behavior must stay identical.

- [ ] **Step 2: Write the architecture map**

Create `brain2/ARCHITECTURE.md` documenting: the layer diagram (core → io → faculties → training → tests), what each package/subpackage is responsible for, the dependency direction rule, and any documented up-edge exceptions from `_refactor/exceptions.md`.

- [ ] **Step 3: Final full gate + regenerate the coupling graph**

Run:
```bash
cd brain2 && ../venv2/bin/python3 _refactor/gate.py; echo "EXIT=$?"
```
Expected: `EXIT=0`. Then re-run the fan-in/out + cycle analysis from the spec exploration to confirm the top-level graph is now ~6 nodes and still acyclic.

- [ ] **Step 4: Remove tooling and commit**

```bash
cd brain2
git rm -r _refactor
# keep baseline/ if you want a permanent regression snapshot; else: git rm -r baseline
git add -A
git commit -m "refactor(brain2): package facades + ARCHITECTURE.md; remove refactor tooling"
```

- [ ] **Step 5: Confirm the whole suite is green from a clean state**

Run: `cd brain2 && for s in harden_regress exam test_open_lang test_phase_a stress_exam; do [ -f $s.py ] && ../venv2/bin/python3 $s.py >/dev/null 2>&1 && echo "ok $s" || echo "check $s"; done`
Expected: `ok` for each suite that was green pre-refactor. Any `check` = investigate before declaring done.

---

## Self-review notes (coverage against spec)

- Layered tree + subpackages → Tasks 4–16.
- Pure re-export facades → Task 2 (`codemod` creates `__init__`), Task 18 (populate).
- Codemod import rewrites, no logic change → Task 2 + every move task.
- Root shims keep invocation + allow-list working → Task 17.
- Full output-diff gate after every move → Tasks 3–17 (`gate.py`).
- Up-edges resolved by relocation/exception → Task 1 Step 3.
- experimental/ quarantine with human approval → Task 16.
- C++ `import brain2` untouched → Task 5 note + not in move_map.
- Determinism/green screen → Task 3.
