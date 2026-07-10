#!/usr/bin/env python3
"""
means_ends.py — Phase 2 + 3: the problem-solving EXECUTIVE with a LEARNED,
STORED policy memory and a conjecture -> verify -> admit loop.

Phase 2 (the spine): Newell & Simon's means-ends analysis over a BLACKBOARD.
  Take a goal. If a source answers directly, done. Otherwise a source that knows
  HOW says what it's MISSING, posts those as sub-goals, recurse, compose. Every
  solved sub-goal is MEMOIZED (tabling). Sources COMMUNICATE through the shared
  blackboard — they read/write it, they do not fuse (the crisp/fuzzy membrane).

Phase 3 (this version):
  * Policies live in a PolicyMemory — a stored, persistable, learnable table,
    SEPARATE from the fact KB. Each policy is a serializable tuple-formula over
    named inputs (evaluated by the real physics_engine.ev). "Rules in memory,
    not hardcoded boards."
  * The PolicyLearner CONJECTURES new policies by composing existing ones
    (inlining a sub-policy), then VERIFIES the conjecture numerically against the
    executive's own step-by-step derivation before ADMITTING it. A wrong
    conjecture is REJECTED at the gate. Induction proposes; verification disposes.

  force = mass*accel ; power = force*speed   --conjecture-->  power = (mass*accel)*speed
  verified against the 2-step derivation -> admitted -> future power queries are
  one shot (the composition is "compiled" into a new stored policy).
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.reasoning.reasoning_engine import ReasoningEngine


def ev(expr, env):
    """Evaluate a tuple-formula over {symbol: value}. Local (not physics_engine's)
    because that one builds the whole op-dict eagerly — it computes a**b even for
    a '*' op, overflowing on large values. This evaluates lazily, one op."""
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, str):
        if expr not in env:
            raise KeyError(f"unknown value for '{expr}'")
        return env[expr]
    op = expr[0]
    if op == "neg":
        return -ev(expr[1], env)
    a = ev(expr[1], env)
    b = ev(expr[2], env)
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b
    if op == "^": return a ** b
    raise ValueError(f"unknown op {op!r}")


@dataclass(frozen=True)
class Need:
    """A goal: the value of `rel` for `subject` (e.g. Need('rocket','force'))."""
    subject: str
    rel: str

    def __str__(self):
        return f"{self.subject}.{self.rel}"


# ── policy memory: stored composition rules, separate from facts ─────────────
@dataclass(frozen=True)
class Policy:
    """target = f(inputs), where expr is a tuple-formula over the input names.
    Serializable (tuples/strings/numbers), evaluated by physics_engine.ev."""
    target: str
    inputs: tuple
    expr: tuple


def _to_jsonable(e):
    return [_to_jsonable(x) for x in e] if isinstance(e, tuple) else e


def _from_jsonable(e):
    return tuple(_from_jsonable(x) for x in e) if isinstance(e, list) else e


class PolicyMemory:
    """A durable, learnable table of policies. The crisp policy store — it talks
    to the fact memory only through the blackboard, it does not merge with it."""
    def __init__(self):
        self.by_target = {}

    def add(self, p: Policy):
        self.by_target[p.target] = p

    def get(self, target):
        return self.by_target.get(target)

    def __contains__(self, target):
        return target in self.by_target

    def save(self, path):
        data = {t: {"inputs": list(p.inputs), "expr": _to_jsonable(p.expr)}
                for t, p in self.by_target.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        m = cls()
        with open(path) as f:
            data = json.load(f)
        for t, d in data.items():
            m.add(Policy(t, tuple(d["inputs"]), _from_jsonable(d["expr"])))
        return m


# ── the blackboard: shared working memory (read/write, never bind) ────────────
class Blackboard:
    def __init__(self):
        self.solved = {}          # Need -> value         (memoization / tabling)
        self.trace = []

    def log(self, msg, depth=0):
        self.trace.append("  " * depth + msg)

    def show(self):
        print("\n".join(self.trace))


# ── knowledge sources: each owns one kind of competence ──────────────────────
class KnowledgeSource(ABC):
    @abstractmethod
    def can_handle(self, need): ...
    @abstractmethod
    def contribute(self, need, solver, depth): ...   # value or None


class FactSource(KnowledgeSource):
    """Crisp fact memory: answers a Need straight from the ReasoningEngine."""
    def __init__(self, kb: ReasoningEngine):
        self.kb = kb

    def can_handle(self, need):
        ans, _ = self.kb.ask(need.subject, need.rel)
        return ans is not None

    def contribute(self, need, solver, depth):
        ans, why = self.kb.ask(need.subject, need.rel)
        if ans is None:
            return None
        solver.bb.log(f"✓ {need} = {ans}   (fact: {why})", depth)
        return _as_number(ans)


class PolicySource(KnowledgeSource):
    """Policy memory: a stored formula computes the target from inputs it fetches
    as sub-goals (means-ends). Reads PolicyMemory — nothing hardcoded here."""
    def __init__(self, mem: PolicyMemory):
        self.mem = mem

    def can_handle(self, need):
        return need.rel in self.mem

    def contribute(self, need, solver, depth):
        p = self.mem.get(need.rel)
        solver.bb.log(f"? {need} not a fact — policy needs {', '.join(p.inputs)}", depth)
        env = {}
        for r in p.inputs:                        # means-ends: fetch what's missing
            v = solver.solve(Need(need.subject, r), depth + 1)
            if v is None:
                solver.bb.log(f"✗ could not get {need.subject}.{r}", depth + 1)
                return None
            env[r] = v
        result = ev(p.expr, env)
        solver.bb.log(f"= {need} = {result}   (policy {need.rel})", depth)
        return result


# ── the executive ────────────────────────────────────────────────────────────
class MeansEndsSolver:
    def __init__(self, sources):
        self.sources = sources
        self.bb = Blackboard()

    def solve(self, need, depth=0):
        if need in self.bb.solved:                # memoization / tabling
            self.bb.log(f"· {need} = {self.bb.solved[need]}   (memo)", depth)
            return self.bb.solved[need]
        for src in self.sources:                  # facts first, then decomposers
            if src.can_handle(need):
                val = src.contribute(need, self, depth)
                if val is not None:
                    self.bb.solved[need] = val
                    return val
        return None


# ── conjecture -> verify -> admit: learn new policies by composition ─────────
def substitute(expr, name, sub):
    """Replace every occurrence of the symbol `name` with `sub` in a formula."""
    if expr == name:
        return sub
    if isinstance(expr, tuple):
        return tuple(substitute(a, name, sub) for a in expr)
    return expr


class PolicyLearner:
    def __init__(self, mem: PolicyMemory):
        self.mem = mem

    def conjecture(self, target):
        """Compose: inline a sub-policy into the target's policy, yielding a new
        policy that skips the intermediate. Returns (Policy, inlined_input)."""
        p = self.mem.get(target)
        if p is None:
            return None
        for i in p.inputs:
            q = self.mem.get(i)                   # an input that is itself derived
            if q is not None:
                new_inputs = tuple(x for x in p.inputs if x != i) + \
                             tuple(x for x in q.inputs if x not in p.inputs)
                return Policy(target, new_inputs, substitute(p.expr, i, q.expr)), i
        return None

    def verify(self, conj: Policy, solver: MeansEndsSolver, entity, tol=1e-6):
        """Evaluate the conjecture one-shot from leaf facts, compare to the
        executive's step-by-step derivation. Equal -> the composition is sound."""
        ref = solver.solve(Need(entity, conj.target))   # ground truth (current policies)
        if ref is None:
            return False, None, ref
        env = {}
        for inp in conj.inputs:
            v = solver.solve(Need(entity, inp))
            if v is None:
                return False, None, ref
            env[inp] = v
        try:
            got = ev(conj.expr, env)
        except Exception:
            return False, None, ref
        return abs(got - ref) < tol, got, ref

    def learn(self, target, solver, entity):
        """Conjecture a composed policy, verify it, admit only if correct."""
        c = self.conjecture(target)
        if c is None:
            return False
        conj, inlined = c
        ok, got, ref = self.verify(conj, solver, entity)
        if ok:
            self.mem.add(conj)                    # admit: overwrite with compiled form
            return True, conj, inlined, got, ref
        return False, conj, inlined, got, ref


def _as_number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


# ── demo ─────────────────────────────────────────────────────────────────────
def _demo():
    kb = ReasoningEngine()
    for s, r, o in [("rocket", "mass", "1000"), ("rocket", "accel", "20"),
                    ("rocket", "speed", "300")]:
        kb.learn(s, r, o)

    mem = PolicyMemory()
    mem.add(Policy("force", ("mass", "accel"), ("*", "mass", "accel")))
    mem.add(Policy("power", ("force", "speed"), ("*", "force", "speed")))  # needs force

    solver = MeansEndsSolver([FactSource(kb), PolicySource(mem)])

    print("=== means_ends — executive + stored policy memory ===\n")
    ans = solver.solve(Need("rocket", "power"))
    print(f">>> rocket.power = {ans}   (derived: power<-force<-mass,accel)\n")

    # ── conjecture -> verify -> admit ────────────────────────────────────────
    print("--- learning: conjecture a composed policy, verify, admit ---")
    learner = PolicyLearner(mem)
    fresh = MeansEndsSolver([FactSource(kb), PolicySource(PolicyMemory_with(mem))])
    ok, conj, inlined, got, ref = learner.learn("power", fresh, "rocket")
    print(f"  conjecture: power = {conj.expr}  (inlined '{inlined}', inputs {conj.inputs})")
    print(f"  verify: one-shot={got} vs step-by-step={ref} -> {'MATCH ✓ admitted' if ok else 'MISMATCH ✗ rejected'}")

    # ── a WRONG conjecture must be rejected at the gate ──────────────────────
    print("\n--- the gate rejects a bad conjecture ---")
    bad = Policy("power", ("mass", "accel", "speed"),
                 ("+", "mass", ("*", "accel", "speed")))   # wrong formula
    okb, gotb, refb = learner.verify(bad, MeansEndsSolver([FactSource(kb), PolicySource(mem)]), "rocket")
    print(f"  bad: power = {bad.expr}")
    print(f"  verify: one-shot={gotb} vs truth={refb} -> {'admitted ✗' if okb else 'REJECTED ✓ (not stored)'}")

    # ── the admitted policy is now one-shot (force step compiled away) ────────
    print("\n--- after learning: power is now one-shot ---")
    s2 = MeansEndsSolver([FactSource(kb), PolicySource(mem)])
    print(f">>> rocket.power = {s2.solve(Need('rocket','power'))}")
    s2.bb.show()

    # persistence: the learned policy survives a save/load round-trip
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "policies.json")
    mem.save(path)
    reloaded = PolicyMemory.load(path)
    print(f"\npersisted -> reloaded power policy: {reloaded.get('power').expr}")


def PolicyMemory_with(src: PolicyMemory):
    """A shallow copy so verification uses the ORIGINAL (uncomposed) policies as
    ground truth, without the conjecture leaking in before it's verified."""
    m = PolicyMemory()
    for t, p in src.by_target.items():
        m.add(p)
    return m


if __name__ == "__main__":
    _demo()
