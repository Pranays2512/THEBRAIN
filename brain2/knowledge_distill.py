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
    arbitrary teacher data). means_ends.FactSource just needs .ask(subj, rel) -> (value, conf).

    MULTI-VALUE: the same (entity, rel) can legitimately carry different values across the
    corpus (e.g. a story's 'pencils | number' is 6 in one grade, 12 in another). A single-slot
    dict lost all but the last (key collision). We keep every distinct value; .ask returns the
    most-recent (compute entities have a single value, so this is unchanged for them), and
    .values / .knows expose the full set so recall isn't silently dropped."""
    def __init__(self):
        self.facts = {}          # (e, r) -> most-recently-learned value (fast path for compute)
        self.multi = {}          # (e, r) -> list of distinct values, in insertion order

    def learn(self, e, r, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            pass
        k = (e.lower(), r.lower())
        self.facts[k] = v
        vals = self.multi.setdefault(k, [])
        if v not in vals:
            vals.append(v)

    def ask(self, subj, rel, **kw):
        return self.facts.get((subj.lower(), rel.lower())), 1.0

    def values(self, subj, rel):
        """All distinct values learned for this key (multi-value recall)."""
        return list(self.multi.get((subj.lower(), rel.lower()), []))

    def knows(self, subj, rel, value):
        """True if `value` is among the values learned for (subj, rel)."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
        return value in self.multi.get((subj.lower(), rel.lower()), [])


_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _desuper(s):
    """Rewrite unicode superscript exponents to caret form: 2¹⁰ -> 2^10."""
    return re.sub("[" + "".join(_SUP) + "]+",
                  lambda m: "^" + "".join(_SUP[c] for c in m.group(0)), s)


def infix_to_tree(s):
    """Parse an infix expression ('mass * accel', '0.5 * m * v', '2^10') to a tree."""
    toks = re.findall(r"[a-zA-Z_]\w*|\d+\.?\d*|[+\-*/^()]", _desuper(s))
    prec = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
    right = {"^"}                              # power is right-associative
    out, ops = [], []
    for t in toks:
        if re.match(r"[a-zA-Z_]", t):
            out.append(t)
        elif re.match(r"\d", t):
            out.append(float(t) if "." in t else int(t))
        elif t in prec:
            while ops and ops[-1] in prec and (
                    prec[ops[-1]] > prec[t]
                    or (prec[ops[-1]] == prec[t] and t not in right)):
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
            e = m.group(1).strip().lower()
            r = m.group(2).strip().lower()
            if e and r:
                facts.append((e, r, m.group(3)))
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
        "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else 0.0,
        "^": lambda a, b: a ** b}

# ── learned arithmetic (grounded in succ/pred), lazily synthesised once ───────
# The brain COMPUTES +,-,* with procedures it learned (math_synth), not host
# operators. Only +,-,* on non-negative integers route to the learned library;
# division / floats / negatives fall back to host (no learned div yet). Every
# op records which path it took, so we can report how much runs on learned math.
_LA = None
_LA_FAILED = False
ARITH_STATS = {"learned": 0, "host": 0}


def _learned_lib():
    global _LA, _LA_FAILED
    if _LA is None and not _LA_FAILED:
        try:
            from math_synth import LearnedArithmetic
            _LA = LearnedArithmetic()
        except Exception:
            _LA_FAILED = True
    return _LA


def reset_arith_stats():
    ARITH_STATS["learned"] = ARITH_STATS["host"] = 0


def _arith(op, a, b):
    """Compute a op b. Route +,-,* on non-neg ints to LEARNED procedures; else host."""
    la = _learned_lib()
    is_nat = (la is not None and float(a).is_integer() and float(b).is_integer()
              and a >= 0 and b >= 0)
    if is_nat:
        ia, ib = int(a), int(b)
        name = {"+": "add", "*": "mul", "^": "pow"}.get(op)
        if op == "-" and ia >= ib:                 # learned sub is truncated (monus)
            name = "sub"
        if name in la.lib:
            try:
                from math_synth import safe_call
                val = safe_call(la.lib[name], ia, ib, budget=5_000_000)
                ARITH_STATS["learned"] += 1
                return float(val)
            except Exception:                      # too big for grounded (WorkExceeded) → host
                pass
    ARITH_STATS["host"] += 1
    return _OPS[op](a, b)


def _eval(tree, env):
    if isinstance(tree, str):
        return env[tree]                    # KeyError if a var isn't a known fact -> caller guards
    if not isinstance(tree, tuple):
        return float(tree)
    return _arith(tree[0], _eval(tree[1], env), _eval(tree[2], env))


def teach(fkb, mem, facts, laws):
    """Teach facts; admit each law as a policy once it VERIFIES on some object. Verification is
    MULTI-HOP: a law's inputs may be direct facts OR quantities DERIVED by already-admitted
    policies (checked through the means-ends solver, which chains + memoizes). So laws admit to
    a FIXPOINT — a law admitted in one pass supplies a derived input that unblocks dependent
    laws in the next. (The old single-level check dropped every law whose input wasn't a raw
    fact, even when the brain could compute it.) A verified law's value is stored so the brain
    'knows' it; still one verifying object is enough to admit the reusable policy."""
    from means_ends import Policy, FactSource, PolicySource, MeansEndsSolver, Need
    ent_facts = {}
    for e, r, v in facts:
        fkb.learn(e, r, v)
        try:
            ent_facts.setdefault(e.lower(), {})[r.lower()] = float(v)
        except (TypeError, ValueError):
            pass
    entities = list(ent_facts.keys())
    admitted, pending = [], list(laws)
    changed = True
    while changed and pending:
        changed = False
        solve = MeansEndsSolver([FactSource(fkb), PolicySource(mem)]).solve   # memoizes per pass
        still = []
        for target, tree in pending:
            inputs = [v for v in _vars(tree) if isinstance(v, str)]
            # only bother with objects that could plausibly supply an input (have ANY fact
            # whose rel is one of the inputs) — keeps the entity scan cheap on a big corpus
            cands = [e for e in entities if any(i in ent_facts[e] for i in inputs)] or entities[:1]
            done = False
            for ent in cands:
                env = {}
                if all((env.setdefault(i, solve(Need(ent, i))) is not None) for i in inputs):
                    try:
                        val = _eval(tree, env)
                    except Exception:
                        continue
                    mem.add(Policy(target, tuple(inputs), tree))
                    fkb.learn(ent, target, val)
                    admitted.append(target)
                    changed = done = True
                    break
            if not done:
                still.append((target, tree))
        pending = still
    return admitted, [t for t, _ in pending]


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
