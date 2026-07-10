#!/usr/bin/env python3
"""domain_features.py — domain knowledge the proposer can use.

Hard filter: dimensional analysis. A dimensionally-inconsistent policy cannot be correct, so
it is pruned BEFORE search (score 0) — pure structure, zero learning, the strongest pruning
signal physics/chem offer. Unknown units -> None (abstain): the filter must NEVER prune what
it does not understand (the three-valued True/False/None contract). Soft feature: per-policy
Laplace-smoothed success rate. (Plan Phase A, Task 10.)"""


class DimError(Exception):
    pass


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _mul(a, k): return tuple(x * k for x in a)


def dims_of(expr, units):
    """Exponent vector of a tuple-formula, or raise DimError on inconsistency, or KeyError on
    an unknown symbol. Numeric scalars are dimensionless (None)."""
    if isinstance(expr, (int, float)):
        return None
    if isinstance(expr, str):
        return units[expr]
    op = expr[0]
    if op == "neg":
        return dims_of(expr[1], units)
    a = dims_of(expr[1], units)
    b = dims_of(expr[2], units)
    if op in ("+", "-"):
        if a != b:
            raise DimError("%s %s %s" % (a, op, b))
        return a
    if op == "*":
        if a is None: return b
        if b is None: return a
        return _add(a, b)
    if op == "/":
        if a is None: a = tuple(0 for _ in (b or ()))
        if b is None: return a
        return _sub(a, b)
    if op == "^":
        if not isinstance(expr[2], (int, float)):
            raise DimError("non-numeric exponent")
        return None if a is None else _mul(a, expr[2])
    raise DimError("unknown op %r" % op)


def dim_consistent(policy, units):
    """True (consistent), False (provably wrong), None (unknown units: abstain, do not prune)."""
    try:
        d = dims_of(policy.expr, units)
    except DimError:
        return False
    except KeyError:
        return None
    target = units.get(policy.target)
    if target is None or d is None:
        return None
    return d == target


def success_rate_feature(policy, history):
    """Laplace-smoothed win rate of this exact policy: (wins+1)/(wins+losses+2)."""
    wins, losses = history.get((policy.target, policy.inputs), (0, 0))
    return (wins + 1) / (wins + losses + 2)
