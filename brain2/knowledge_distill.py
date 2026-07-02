#!/usr/bin/env python3
"""
knowledge_distill.py — the cloud AI PARSES data into both halves of the brain.

The teacher (qwen-coder) is used not just to make corpus text, but to PARSE a domain into
STRUCTURED knowledge the symbolic brain can ingest and VERIFY — plus sentences the student LM
learns. One teacher, both halves:

  teacher emits, per topic, a strict format:
    OBJECT: <name>
    FACT:  <name> | <property> | <number>
    LAW:   <derived> = <expression over the properties, using + - * / and numbers>
    SENT:  <one plain factual sentence>

  -> FACTs  -> taught to the symbolic reasoner (ReasoningEngine.learn)
  -> LAWs   -> parsed to a policy, VERIFIED (must compute on the object's facts) then ADMITTED
  -> SENTs  -> corpus -> the student LM learns from the same knowledge

Membrane holds: the teacher PROPOSES facts/laws; the brain VERIFIES a law computes before
admitting it (a law that doesn't evaluate on the facts is rejected). The student learns the
language; the symbolic core learns the verified structure. Both from one parsed source.
"""

import re


class SimpleKB:
    """Exact fact store for taught facts (avoids the fuzzy ReasoningEngine's recursion on
    arbitrary teacher data). means_ends.FactSource just needs .ask(subj, rel) -> (value, conf)."""
    def __init__(self):
        self.facts = {}

    def learn(self, e, r, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            pass
        self.facts[(e.lower(), r.lower())] = v

    def ask(self, subj, rel, **kw):
        return self.facts.get((subj.lower(), rel.lower())), 1.0


def infix_to_tree(s):
    """Parse an infix expression ('mass * accel', '0.5 * m * v') to a nested-tuple tree."""
    toks = re.findall(r"[a-zA-Z_]\w*|\d+\.?\d*|[+\-*/()]", s)
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    out, ops = [], []
    for t in toks:
        if re.match(r"[a-zA-Z_]", t):
            out.append(t)
        elif re.match(r"\d", t):
            out.append(float(t) if "." in t else int(t))
        elif t in prec:
            while ops and ops[-1] in prec and prec[ops[-1]] >= prec[t]:
                out.append(ops.pop())
            ops.append(t)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if ops:
                ops.pop()
    while ops:
        out.append(ops.pop())
    st = []
    for t in out:
        if t in prec:
            if len(st) < 2:
                return None
            b, a = st.pop(), st.pop()
            st.append((t, a, b))
        else:
            st.append(t)
    return st[0] if len(st) == 1 else None


def _vars(tree):
    if isinstance(tree, str):
        return {tree}
    if not isinstance(tree, tuple):
        return set()
    out = set()
    for k in tree[1:]:
        out |= _vars(k)
    return out


def parse_teacher(text):
    """Extract (objects, facts, laws, sents) from the teacher's structured output."""
    facts, laws, sents = [], [], []
    cur_obj = None
    for ln in text.splitlines():
        ln = ln.strip()
        m = re.match(r"OBJECT:\s*(.+)", ln)
        if m:
            cur_obj = re.findall(r"[a-z_]+", m.group(1).lower())[:1]
            cur_obj = cur_obj[0] if cur_obj else None
            continue
        m = re.match(r"FACT:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(-?\d+\.?\d*)", ln)
        if m:
            e = re.findall(r"[a-z_]+", m.group(1).lower())
            r = re.findall(r"[a-z_]+", m.group(2).lower())
            if e and r:
                facts.append((e[0], r[0], m.group(3)))
            continue
        m = re.match(r"LAW:\s*([a-zA-Z_]\w*)\s*=\s*(.+)", ln)
        if m:
            tree = infix_to_tree(m.group(2))
            if tree is not None:
                laws.append((m.group(1).lower(), tree))
            continue
        m = re.match(r"SENT:\s*(.+)", ln)
        if m and len(m.group(1).split()) >= 4:
            sents.append(m.group(1).strip())
    return facts, laws, sents


_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else 0.0}


def _eval(tree, env):
    if isinstance(tree, str):
        return env[tree]                    # KeyError if a var isn't a known fact -> caller guards
    if not isinstance(tree, tuple):
        return float(tree)
    return _OPS[tree[0]](_eval(tree[1], env), _eval(tree[2], env))


def teach(fkb, mem, facts, laws):
    """Teach facts; admit each law as a policy only if it VERIFIES by DIRECT evaluation on an
    object whose facts cover its inputs (single-level, so no policy-chain recursion/cycles).
    A verified law's derived value is also stored, so the brain 'knows' it."""
    from means_ends import Policy
    ent_facts = {}
    for e, r, v in facts:
        fkb.learn(e, r, v)
        try:
            ent_facts.setdefault(e.lower(), {})[r.lower()] = float(v)
        except (TypeError, ValueError):
            pass
    admitted, rejected = [], []
    for target, tree in laws:
        inputs = [v for v in _vars(tree) if isinstance(v, str)]
        ent = next((e for e, fs in ent_facts.items() if all(i in fs for i in inputs)), None)
        if ent is None:
            rejected.append(target)         # inputs aren't known facts -> unverifiable
            continue
        try:
            val = _eval(tree, ent_facts[ent])
        except Exception:
            rejected.append(target)
            continue
        mem.add(Policy(target, tuple(inputs), tree))
        fkb.learn(ent, target, val)          # store the derived value
        admitted.append(target)
    return admitted, rejected


# ── offline self-test of the parser + teach + verify (no teacher calls) ──
def _self_test():
    from means_ends import PolicyMemory
    mock = """
OBJECT: capacitor
FACT: capacitor | charge | 12
FACT: capacitor | voltage | 4
FACT: capacitor | area | 3
LAW: capacitance = charge / voltage
LAW: bogus = charge * missing_thing
SENT: a capacitor stores electric charge on two plates
SENT: capacitance is the ratio of charge to voltage
"""
    facts, laws, sents = parse_teacher(mock)
    fkb, mem = SimpleKB(), PolicyMemory()
    adm, rej = teach(fkb, mem, facts, laws)
    from means_ends import FactSource, PolicySource, MeansEndsSolver, Need
    solve = MeansEndsSolver([FactSource(fkb), PolicySource(mem)]).solve
    print("=== knowledge_distill self-test (offline, mock teacher output) ===")
    print("  parsed: %d facts, %d laws, %d sents" % (len(facts), len(laws), len(sents)))
    print("  admitted (verified) laws:", adm, " rejected:", rej)
    print("  capacitor.capacitance =", solve(Need("capacitor", "capacitance")), "(12/4 = 3, computed)")
    print("  -> teacher-parsed knowledge taught + verified; the bogus law (undefined var) rejected.")


if __name__ == "__main__":
    _self_test()
