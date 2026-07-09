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
