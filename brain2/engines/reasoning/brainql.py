#!/usr/bin/env python3
"""
brainql.py — BrainQL: the structured query language for LLM↔Brain communication.

The LLM is ONLY eyes and mouth. BrainQL is the protocol between them:
  - LLM generates BrainQL from natural language  (Eyes)
  - Brain executes BrainQL deterministically      (Thinker)
  - LLM verbalizes BrainQLResult into text        (Mouth)

The LLM never reasons. It only translates TO and FROM BrainQL.
The Brain never sees raw text (beyond what it needs for grounding).

Language has 8 operations:

  LOOKUP   <subj> <rel>                   → direct fact lookup
  CHAIN    <subj> <rel> [hops=N]         → transitive closure (isa chains)
  INHERIT  <subj> <rel>                   → property inherited via isa hierarchy
  DERIVE   <subj> <rel>                   → full composition rule application
  TEACH    <subj> <rel> <obj>            → assert a new fact
  TEACH_RULE <prem1> <prem2> -> <concl> → assert a new inference rule
  COMPUTE  <entity> <property>           → means-ends solver (physics/math)
  EXPLAIN  <subj> <rel>                  → return proof chain (not just answer)

Usage:
    from engines.reasoning.brainql import parse_bql, BrainQLResult

    q = parse_bql("INHERIT HCL turns_litmus")
    # → BrainQLQuery(op='INHERIT', subj='HCL', rel='turns_litmus')

    # Execute with a ReasoningEngine:
    from engines.reasoning.brainql import BrainQLExecutor
    result = BrainQLExecutor(reasoning_engine).run(q)
    # → BrainQLResult(op='INHERIT', subj='HCL', rel='turns_litmus',
    #                 value='red', chain=['HCL isa acid', 'acid turns_litmus red'],
    #                 verified=True)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any

# ── IR types ────────────────────────────────────────────────────────────────

VALID_OPS = {"LOOKUP", "CHAIN", "INHERIT", "DERIVE", "TEACH", "TEACH_RULE", "COMPUTE", "EXPLAIN"}


@dataclass
class BrainQLQuery:
    """One BrainQL instruction as parsed from text.

    Fields that are not used by a particular op are left as ''.
    """
    op: str          # one of VALID_OPS
    subj: str = ""   # primary subject token
    rel: str = ""    # relation token
    obj: str = ""    # object token (TEACH only)
    prem1: str = ""  # first premise (TEACH_RULE only)
    prem2: str = ""  # second premise (TEACH_RULE only)
    concl: str = ""  # conclusion (TEACH_RULE only)
    hops: int = 8    # max depth for CHAIN / DERIVE
    raw: str = ""    # the original text line

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in ("", 8) or k == "hops"}


@dataclass
class BrainQLResult:
    """The brain's structured answer to a BrainQLQuery.

    The LLM receives this JSON-serialisable object and verbalises it.
    It NEVER reasons over it — it only renders it into fluent text.
    """
    op: str
    subj: str = ""
    rel: str = ""
    obj: str = ""         # echo of the taught object (TEACH)
    value: Any = None     # the answer: a string, number, list, or None
    chain: list = field(default_factory=list)   # human-readable derivation steps
    verified: bool = False
    known: bool = False
    note: str = ""        # honest abstention message when known=False

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "subj": self.subj,
            "rel": self.rel,
            "value": self.value,
            "chain": self.chain,
            "verified": self.verified,
            "known": self.known,
            "note": self.note,
        }

    def __repr__(self) -> str:
        if self.known:
            return (f"BrainQLResult({self.op} {self.subj} {self.rel} "
                    f"→ {self.value!r} chain={self.chain} verified={self.verified})")
        return f"BrainQLResult({self.op} {self.subj} {self.rel} → UNKNOWN: {self.note})"


# ── Parser ───────────────────────────────────────────────────────────────────

class BrainQLParseError(ValueError):
    pass


def parse_bql(text: str) -> BrainQLQuery:
    """Parse one BrainQL instruction line.

    Strips comments (# ...) and leading/trailing whitespace.
    Raises BrainQLParseError on unknown ops or malformed lines.

    Examples:
        parse_bql("LOOKUP  acid  turns_litmus")
        parse_bql("INHERIT HCL   turns_litmus")
        parse_bql("TEACH   HCL   isa  acid")
        parse_bql("TEACH_RULE isa turns_litmus -> turns_litmus")
        parse_bql("CHAIN   HCL isa hops=4")
        parse_bql("COMPUTE rocket force")
        parse_bql("EXPLAIN HCL turns_litmus")
    """
    # strip inline comments
    line = text.split("#")[0].strip()
    if not line:
        raise BrainQLParseError("empty instruction")

    parts = line.split()
    op = parts[0].upper()
    if op not in VALID_OPS:
        raise BrainQLParseError(f"unknown op '{op}'. Valid ops: {sorted(VALID_OPS)}")

    q = BrainQLQuery(op=op, raw=text)

    if op == "TEACH_RULE":
        # TEACH_RULE prem1 prem2 -> concl
        try:
            arrow_idx = parts.index("->")
        except ValueError:
            raise BrainQLParseError("TEACH_RULE requires '->' e.g. TEACH_RULE isa turns_litmus -> turns_litmus")
        if arrow_idx < 3:
            raise BrainQLParseError("TEACH_RULE requires two premises before '->'")
        q.prem1 = parts[1]
        q.prem2 = parts[2]
        q.concl = parts[arrow_idx + 1] if arrow_idx + 1 < len(parts) else ""
        if not q.concl:
            raise BrainQLParseError("TEACH_RULE: missing conclusion after '->'")
        return q

    if op == "TEACH":
        # TEACH subj rel obj
        if len(parts) < 4:
            raise BrainQLParseError("TEACH requires: TEACH <subj> <rel> <obj>")
        q.subj, q.rel, q.obj = parts[1], parts[2], parts[3]
        return q

    # All remaining ops: subj rel [hops=N]
    if len(parts) < 3:
        raise BrainQLParseError(f"{op} requires at least: {op} <subj> <rel>")
    q.subj, q.rel = parts[1], parts[2]

    # optional hops= parameter
    for p in parts[3:]:
        m = re.match(r"hops=(\d+)", p)
        if m:
            q.hops = int(m.group(1))

    return q


def parse_bql_block(text: str) -> list[BrainQLQuery]:
    """Parse multiple BrainQL instructions (one per line, '#' comments allowed).

    Skips blank lines and comment-only lines. Returns a list in document order.
    Raises BrainQLParseError on the first bad line.
    """
    queries = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        try:
            queries.append(parse_bql(raw))
        except BrainQLParseError as e:
            raise BrainQLParseError(f"line {lineno}: {e}") from e
    return queries


# ── Executor ─────────────────────────────────────────────────────────────────

class BrainQLExecutor:
    """Execute BrainQL queries against a ReasoningEngine.

    This is the Brain's query interface. The LLM never calls this directly —
    it only generates BrainQL text. WholeBrain instantiates this and calls run().

    Args:
        reasoning_engine: a ReasoningEngine instance.
        means_ends_solver: optional MeansEndsSolver for COMPUTE queries.
    """

    def __init__(self, reasoning_engine, means_ends_solver=None):
        self.re = reasoning_engine
        self.mes = means_ends_solver   # set by WholeBrain for COMPUTE

    def run(self, q: BrainQLQuery) -> BrainQLResult:
        """Dispatch one BrainQL instruction and return a BrainQLResult."""
        try:
            if q.op == "LOOKUP":
                return self._lookup(q)
            if q.op == "CHAIN":
                return self._chain(q)
            if q.op == "INHERIT":
                return self._inherit(q)
            if q.op == "DERIVE":
                return self._derive(q)
            if q.op == "TEACH":
                return self._teach(q)
            if q.op == "TEACH_RULE":
                return self._teach_rule(q)
            if q.op == "COMPUTE":
                return self._compute(q)
            if q.op == "EXPLAIN":
                return self._explain(q)
        except Exception as exc:
            return BrainQLResult(op=q.op, subj=q.subj, rel=q.rel,
                                 known=False, note=f"executor error: {exc}")
        return BrainQLResult(op=q.op, known=False, note=f"unknown op: {q.op}")

    def run_block(self, queries: list[BrainQLQuery]) -> list[BrainQLResult]:
        """Run multiple BrainQL queries in sequence. TEACH/TEACH_RULE side-effects
        are visible to later queries in the same block."""
        return [self.run(q) for q in queries]

    # ── individual op handlers ────────────────────────────────────────────────

    def _lookup(self, q: BrainQLQuery) -> BrainQLResult:
        obj, why = self.re.ask(q.subj, q.rel)
        if obj is None:
            return BrainQLResult(op="LOOKUP", subj=q.subj, rel=q.rel,
                                 known=False, note=f"no direct fact for ({q.subj}, {q.rel})")
        return BrainQLResult(op="LOOKUP", subj=q.subj, rel=q.rel,
                             value=obj, chain=[why] if why else [], known=True, verified=True)

    def _chain(self, q: BrainQLQuery) -> BrainQLResult:
        """Transitive closure of rel from subj."""
        ancestors = self.re.closure(q.subj, q.rel)
        if not ancestors:
            return BrainQLResult(op="CHAIN", subj=q.subj, rel=q.rel,
                                 known=False, note=f"no {q.rel}-ancestors of {q.subj}")
        chain_steps = [" → ".join(path) for path in ancestors.values()]
        return BrainQLResult(op="CHAIN", subj=q.subj, rel=q.rel,
                             value=sorted(ancestors.keys()),
                             chain=chain_steps, known=True, verified=True)

    def _inherit(self, q: BrainQLQuery) -> BrainQLResult:
        """Inherited property: walk isa hierarchy, then look up rel.

        This is the key op for "HCL turns_litmus ?" — it:
        1. Chains isa from subj to find all parent categories
        2. For each parent, does a direct LOOKUP of rel
        3. Returns the first (closest ancestor) hit
        """
        # Step 1: find isa ancestors (if isa is transitive, closure handles multi-hop)
        self.re.set_transitive("isa")
        isa_closure = self.re.closure(q.subj, "isa")

        # Step 2: also check the direct rel on subj itself (may own the fact)
        obj_direct, why_direct = self.re.ask(q.subj, q.rel)
        if obj_direct is not None:
            return BrainQLResult(
                op="INHERIT", subj=q.subj, rel=q.rel,
                value=obj_direct,
                chain=[f"{q.subj} {q.rel} {obj_direct}  (direct)"],
                known=True, verified=True,
            )

        # Step 3: walk up the isa chain (BFS order = nearest ancestor first)
        from collections import deque
        visited = set()
        frontier = deque([[q.subj]])
        while frontier:
            path = frontier.popleft()
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            obj, why = self.re.ask(node, q.rel)
            if obj is not None:
                isa_steps = [f"{path[i]} isa {path[i+1]}" for i in range(len(path)-1)]
                return BrainQLResult(
                    op="INHERIT", subj=q.subj, rel=q.rel,
                    value=obj,
                    chain=isa_steps + [f"{node} {q.rel} {obj}"],
                    known=True, verified=True,
                )
            # expand to isa-parents of this node
            parents = sorted(self.re.ask_all(node, "isa").keys())
            for p in parents:
                if p not in visited:
                    frontier.append(path + [p])

        return BrainQLResult(op="INHERIT", subj=q.subj, rel=q.rel,
                             known=False,
                             note=f"no {q.rel} found for {q.subj} or its isa ancestors")

    def _derive(self, q: BrainQLQuery) -> BrainQLResult:
        """Full composition-rule derivation."""
        obj, why = self.re.ask(q.subj, q.rel, max_depth=q.hops)
        if obj is None:
            return BrainQLResult(op="DERIVE", subj=q.subj, rel=q.rel,
                                 known=False,
                                 note=f"could not derive ({q.subj}, {q.rel}) with current rules")
        return BrainQLResult(op="DERIVE", subj=q.subj, rel=q.rel,
                             value=obj, chain=[why] if why else [], known=True, verified=True)

    def _teach(self, q: BrainQLQuery) -> BrainQLResult:
        """Assert a new fact and return confirmation."""
        was_new = self.re.learn(q.subj, q.rel, q.obj)
        return BrainQLResult(
            op="TEACH", subj=q.subj, rel=q.rel, obj=q.obj,
            value=q.obj, known=True, verified=True,
            note="new" if was_new else "already known",
        )

    def _teach_rule(self, q: BrainQLQuery) -> BrainQLResult:
        """Assert a new composition rule."""
        self.re.add_rule(q.prem1, q.prem2, q.concl)
        return BrainQLResult(
            op="TEACH_RULE", known=True, verified=True,
            value=f"{q.prem1} ∘ {q.prem2} → {q.concl}",
            chain=[f"rule: X {q.prem1} Y AND Y {q.prem2} Z => X {q.concl} Z"],
        )

    def _compute(self, q: BrainQLQuery) -> BrainQLResult:
        """Means-ends physics/math computation."""
        if self.mes is None:
            return BrainQLResult(op="COMPUTE", subj=q.subj, rel=q.rel,
                                 known=False, note="no MeansEndsSolver available for COMPUTE")
        try:
            from engines.reasoning.means_ends import Need, FactSource, PolicySource
            v = self.mes.solve(Need(q.subj, q.rel))
        except Exception as exc:
            return BrainQLResult(op="COMPUTE", subj=q.subj, rel=q.rel,
                                 known=False, note=f"solver error: {exc}")
        if v is None:
            return BrainQLResult(op="COMPUTE", subj=q.subj, rel=q.rel,
                                 known=False, note=f"cannot compute {q.rel} of {q.subj} from known facts")
        return BrainQLResult(op="COMPUTE", subj=q.subj, rel=q.rel,
                             value=v, known=True, verified=True,
                             chain=[f"{q.subj}.{q.rel} = {v:.4g}" if isinstance(v, float) else f"{q.subj}.{q.rel} = {v}"])

    def _explain(self, q: BrainQLQuery) -> BrainQLResult:
        """Like DERIVE but returns the full proof chain regardless of depth."""
        obj, why = self.re.ask(q.subj, q.rel, max_depth=q.hops)
        if obj is None:
            return BrainQLResult(op="EXPLAIN", subj=q.subj, rel=q.rel,
                                 known=False, note=f"nothing to explain: ({q.subj}, {q.rel}) is unknown")
        # Also collect isa ancestry for context
        isa_ancestors = list(self.re.closure(q.subj, "isa").keys())
        chain = [why] if why else []
        if isa_ancestors:
            chain.append(f"isa-ancestors of {q.subj}: {', '.join(isa_ancestors)}")
        return BrainQLResult(op="EXPLAIN", subj=q.subj, rel=q.rel,
                             value=obj, chain=chain, known=True, verified=True)


# ── LLM prompt helpers ───────────────────────────────────────────────────────

EYES_SYSTEM_BQL = """\
You convert natural language into BrainQL — a structured query language for a symbolic reasoning engine.
Output ONLY BrainQL instructions, one per line. Do NOT answer the question yourself.

BrainQL operations:
  LOOKUP  <subj> <rel>                   # direct fact: what is subj's rel?
  CHAIN   <subj> <rel>                   # transitive: all ancestors via rel
  INHERIT <subj> <rel>                   # inherited from isa hierarchy
  DERIVE  <subj> <rel>                   # full inference with composition rules
  TEACH   <subj> <rel> <obj>            # assert new fact
  TEACH_RULE <p1> <p2> -> <concl>       # assert new rule: X p1 Y AND Y p2 Z => X concl Z
  COMPUTE <entity> <property>           # physics/math: compute a quantity
  EXPLAIN <subj> <rel>                  # show proof chain

Rules:
- Use lowercase_underscore tokens (e.g. turns_litmus, HCL → hcl)
- Questions about inheritance/properties: use INHERIT
- Questions about causal chains: use CHAIN
- Teaching a fact ("X is a Y", "X can Z"): use TEACH
- Teaching a rule: use TEACH_RULE
- Math/physics quantities: use COMPUTE
- "why" / "how do you know": use EXPLAIN
- If the question involves multiple steps: emit multiple BrainQL lines

Examples:
  "Does HCL turn litmus red?"          → INHERIT hcl turns_litmus
  "What is apple?"                     → DERIVE apple isa
  "A dog is an animal"                 → TEACH dog isa animal
  "All acids turn litmus red"          → TEACH acid turns_litmus red
  "What is the force of the rocket?"   → COMPUTE rocket force
  "How do you know HCL turns litmus?"  → EXPLAIN hcl turns_litmus
  "Is a cat a mammal?"                 → CHAIN cat isa
  "What can a bird do?"                → INHERIT bird can
"""


MOUTH_SYSTEM_BQL = """\
You render a BrainQL result into one natural, fluent English sentence.
You receive a JSON object with the brain's answer. Use ONLY the facts in the JSON.
Do NOT add any outside information. If known=false, say you don't know honestly.
Output ONLY the sentence — no JSON, no preamble.

Example inputs/outputs:
  {"op":"INHERIT","subj":"hcl","rel":"turns_litmus","value":"red","chain":["hcl isa acid","acid turns_litmus red"],"known":true}
  → "HCL turns litmus red because it is an acid, and all acids turn litmus red."

  {"op":"COMPUTE","subj":"rocket","rel":"force","value":150.0,"known":true}
  → "The force of the rocket is 150."

  {"op":"CHAIN","subj":"dog","rel":"isa","value":["animal","mammal","pet"],"known":true}
  → "A dog is an animal, a mammal, and a pet."

  {"op":"LOOKUP","known":false,"note":"no direct fact for (hcl, colour)"}
  → "I don't know the colour of HCL."
"""


def result_to_mouth_payload(result: BrainQLResult) -> str:
    """Serialise a BrainQLResult as the JSON payload the LLM Mouth receives."""
    import json
    return json.dumps(result.to_dict(), ensure_ascii=False)


# ── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    print("=== BrainQL demo ===\n")

    # 1. Parse examples
    cases = [
        "INHERIT hcl turns_litmus",
        "CHAIN   dog isa",
        "TEACH   hcl isa acid",
        "TEACH_RULE isa turns_litmus -> turns_litmus",
        "COMPUTE rocket force",
        "EXPLAIN hcl turns_litmus",
        "LOOKUP  acid turns_litmus",
    ]
    for c in cases:
        q = parse_bql(c)
        print(f"  parse: {c!r}")
        print(f"         → {q}\n")

    # 2. End-to-end: teach acid, HCL; derive that HCL turns litmus red
    print("--- Generalization test (HCL → acid → turns_litmus → red) ---")
    try:
        from engines.reasoning.reasoning_engine import ReasoningEngine
        re = ReasoningEngine()
        exec_bql = BrainQLExecutor(re)

        # Teach facts
        exec_bql.run(parse_bql("TEACH acid turns_litmus red"))
        exec_bql.run(parse_bql("TEACH HCL isa acid"))

        # Query via INHERIT (the op that walks the isa chain)
        result = exec_bql.run(parse_bql("INHERIT HCL turns_litmus"))
        print(f"  INHERIT HCL turns_litmus → {result}")
        assert result.known, "FAIL: should have found 'red' via isa chain"
        assert result.value == "red", f"FAIL: expected 'red', got {result.value!r}"
        print("  ✓ Generalization works: HCL turns litmus red (via isa acid)\n")
    except ImportError as e:
        print(f"  (skipping live test: {e})\n")

    print("=== BrainQL parser / IR demo passed ===")


if __name__ == "__main__":
    _demo()
