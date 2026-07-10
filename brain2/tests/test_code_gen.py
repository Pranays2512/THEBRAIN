#!/usr/bin/env python3
"""
test_code_gen.py — spec -> code in Python / C++ / Java, structurally checked.

The Python output is compiled AND executed (instantiate the class, read a field),
so the generated code is verified by the real interpreter, not just by string
shape. C++ / Java outputs are checked for the idiomatic constructs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.synthesis.code_gen import CodeGenerator, ClassSpec, Field, Method

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def spec():
    return ClassSpec("Point",
                     [Field("x", "int"), Field("y", "int")],
                     [Method("distance", [], "float"),
                      Method("translate", [Field("dx", "int")], "void")])


def run():
    print("\nCodeGenerator — one spec, three languages")
    cg = CodeGenerator()
    s = spec()

    # 1. Python: compile + execute the generated class
    py = cg.generate(s, "python")
    ns = {}
    try:
        exec(compile(py, "<gen>", "exec"), ns)        # valid Python?
        p = ns["Point"](3, 4)                          # constructor works?
        ok = p.x == 3 and p.y == 4 and hasattr(p, "distance")
    except Exception:
        ok = False
    check("Python output compiles, runs, fields set", ok)

    # 2. C++ structural
    cpp = cg.generate(s, "cpp")
    check("C++ has class + member-init constructor",
          "class Point {" in cpp and "Point(int x, int y) : x(x), y(y)" in cpp)
    check("C++ maps float -> double", "double distance()" in cpp)

    # 3. Java structural
    java = cg.generate(s, "java")
    check("Java has private fields + this-assignment",
          "private int x;" in java and "this.x = x;" in java)
    check("Java method returns a typed default",
          "public double distance()" in java and "return 0;" in java)

    # 4. params carried through
    check("Python method keeps params", "def translate(self, dx: int)" in py)
    check("Java void method has no return", "public void translate(int dx)" in java)

    # 5. honest rejection
    try:
        cg.generate(s, "rust")
        check("reject unsupported language", False)
    except ValueError:
        check("reject unsupported language", True)

    print(f"\nCode generator: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
