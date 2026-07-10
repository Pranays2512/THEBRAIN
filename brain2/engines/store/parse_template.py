#!/usr/bin/env python3
"""parse_template.py — symbolic sentence templates: the crisp grammar unit.

A Template is to language what a Policy is to physics: a stored, serializable, verifiable
rule. ("w", word) items must match exactly; ("slot", name, type) items bind a typed value;
("any",) skips one token. Matching is exact-length — no fuzzy scoring here (the membrane:
fuzz lives in the grounder). Induction (slot_example + anti_unify) mirrors factorizer's move
for formulas, applied to word sequences. (Plan Phase A, Tasks 1/2/4.)"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    rel: str
    items: tuple


# suffix rules: (suffix, replacement) tried in order; first hit wins. Data, not code.
_SUFFIX_RULES = (("ing", ""), ("ies", "y"), ("ied", "y"), ("es", "e"), ("ed", ""), ("s", ""))


def _is_number(tok):
    return re.fullmatch(r"\d+(?:\.\d+)?", tok) is not None


def normalize(tok):
    if _is_number(tok) or len(tok) <= 3:
        return tok
    for suf, rep in _SUFFIX_RULES:
        if tok.endswith(suf) and len(tok) - len(suf) + len(rep) >= 3:
            return tok[: len(tok) - len(suf)] + rep
    return tok


def tokenize(text, normalized=False):
    toks = re.findall(r"[a-z_]+|\d+(?:\.\d+)?", text.lower())
    return [normalize(t) for t in toks] if normalized else toks


def match(template, tokens, entities):
    if len(tokens) != len(template.items):
        return None
    bind = {}
    for item, tok in zip(template.items, tokens):
        kind = item[0]
        if kind == "w":
            if tok != item[1]:
                return None
        elif kind == "any":
            continue
        else:                                   # ("slot", name, type)
            _, name, typ = item
            if typ == "number":
                if not _is_number(tok):
                    return None
                bind[name] = float(tok)
            elif typ == "entity":
                if tok not in entities:
                    return None
                bind[name] = tok
    return bind


def slot_example(sentence, label):
    """One labeled example -> maximally literal template: the entity token and the value token
    become slots; every other token stays a literal. Templates stored in normal form."""
    toks = tokenize(sentence, normalized=True)
    items = []
    for tok in toks:
        if tok == normalize(label["entity"]):
            items.append(("slot", "entity", "entity"))
        elif _is_number(tok) and float(tok) == float(label["value"]):
            items.append(("slot", "value", "number"))
        else:
            items.append(("w", tok))
    return Template(rel=label["rel"], items=tuple(items))


def anti_unify(t1, t2):
    """Generalize two templates: agreeing positions stay, disagreeing literals become
    ('any',). Different rel or length -> not the same rule, refuse."""
    if t1.rel != t2.rel or len(t1.items) != len(t2.items):
        return None
    items = []
    for a, b in zip(t1.items, t2.items):
        if a == b:
            items.append(a)
        elif a[0] == "w" and b[0] == "w":
            items.append(("any",))
        else:
            return None            # slot vs literal disagreement: structurally different
    return Template(rel=t1.rel, items=tuple(items))
