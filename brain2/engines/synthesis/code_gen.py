#!/usr/bin/env python3
"""
code_gen.py — generate class boilerplate from a structured spec (Py / C++ / Java).

The honest reading of "build something from proper instructions": a structured
spec (class name, typed fields, method signatures) is PRODUCED into idiomatic
code per language — the same grammar-production idea as the sentence generator,
applied to code. Deterministic and correct by construction for the supported
constructs; it is not logic synthesis and does not write method bodies.

    spec = ClassSpec("Point", [Field("x","int"), Field("y","int")],
                     [Method("distance", [], "float")])
    CodeGenerator().generate(spec, "python")   # -> a valid Python class

Honest scope: OOP scaffolding (fields, constructor, method stubs) across the
three languages + their type mappings. Method LOGIC is left as a stub — writing
arbitrary algorithm bodies is the LLM/synthesis frontier, not template production.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


@dataclass
class Field:
    name: str
    type: str


@dataclass
class Method:
    name: str
    params: list      # list of Field (name, type)
    ret: str = "void"


@dataclass
class ClassSpec:
    name: str
    fields: list = field(default_factory=list)
    methods: list = field(default_factory=list)


# generic type -> language type
TYPES = {
    "python": {"int": "int", "float": "float", "string": "str", "bool": "bool", "void": "None"},
    "cpp":    {"int": "int", "float": "double", "string": "std::string", "bool": "bool", "void": "void"},
    "java":   {"int": "int", "float": "double", "string": "String", "bool": "boolean", "void": "void"},
}
JAVA_DEFAULT = {"int": "0", "double": "0", "boolean": "false", "String": "null"}


def _t(lang, generic):
    return TYPES[lang].get(generic, generic)


class CodeGenerator:
    def generate(self, spec, lang):
        if lang not in TYPES:
            raise ValueError(f"unsupported language: {lang}")
        return getattr(self, f"_{lang}")(spec)

    # ── Python ───────────────────────────────────────────────────────────────
    def _python(self, s):
        L = [f"class {s.name}:"]
        if s.fields:
            args = ", ".join(f"{f.name}: {_t('python', f.type)}" for f in s.fields)
            L.append(f"    def __init__(self, {args}):")
            L += [f"        self.{f.name} = {f.name}" for f in s.fields]
        else:
            L.append("    def __init__(self):")
            L.append("        pass")
        for m in s.methods:
            ps = "".join(f", {p.name}: {_t('python', p.type)}" for p in m.params)
            L.append("")
            L.append(f"    def {m.name}(self{ps}) -> {_t('python', m.ret)}:")
            L.append("        pass")
        return "\n".join(L) + "\n"

    # ── C++ ──────────────────────────────────────────────────────────────────
    def _cpp(self, s):
        L = [f"class {s.name} {{", "public:"]
        L += [f"    {_t('cpp', f.type)} {f.name};" for f in s.fields]
        if s.fields:
            args = ", ".join(f"{_t('cpp', f.type)} {f.name}" for f in s.fields)
            init = ", ".join(f"{f.name}({f.name})" for f in s.fields)
            L.append(f"    {s.name}({args}) : {init} {{}}")
        for m in s.methods:
            ps = ", ".join(f"{_t('cpp', p.type)} {p.name}" for p in m.params)
            ret = _t("cpp", m.ret)
            L.append(f"    {ret} {m.name}({ps}) {{")
            L.append("        // TODO")
            if ret != "void":
                L.append(f"        return {{}};")
            L.append("    }")
        L.append("};")
        return "\n".join(L) + "\n"

    # ── Java ─────────────────────────────────────────────────────────────────
    def _java(self, s):
        L = [f"public class {s.name} {{"]
        L += [f"    private {_t('java', f.type)} {f.name};" for f in s.fields]
        if s.fields:
            args = ", ".join(f"{_t('java', f.type)} {f.name}" for f in s.fields)
            L.append(f"    public {s.name}({args}) {{")
            L += [f"        this.{f.name} = {f.name};" for f in s.fields]
            L.append("    }")
        for m in s.methods:
            ps = ", ".join(f"{_t('java', p.type)} {p.name}" for p in m.params)
            ret = _t("java", m.ret)
            L.append(f"    public {ret} {m.name}({ps}) {{")
            L.append("        // TODO")
            if ret != "void":
                L.append(f"        return {JAVA_DEFAULT.get(ret, 'null')};")
            L.append("    }")
        L.append("}")
        return "\n".join(L) + "\n"


def _demo():
    spec = ClassSpec("Point",
                     [Field("x", "int"), Field("y", "int")],
                     [Method("distance", [], "float"),
                      Method("translate", [Field("dx", "int"), Field("dy", "int")], "void")])
    cg = CodeGenerator()
    print("=== code_gen — one spec, three languages ===\n")
    for lang in ("python", "cpp", "java"):
        print(f"--- {lang} ---")
        print(cg.generate(spec, lang))


if __name__ == "__main__":
    _demo()
