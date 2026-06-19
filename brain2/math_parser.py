#!/usr/bin/env python3
"""
math_parser.py — parse math notation into the engines' expression tuples.

Math notation is a formal grammar, so this is an EXACT recursive-descent parser
(not the controlled-approximation parsing that natural language needs). It turns
"sin(x^2)" / "2*x + 3 = 7" into the nested tuples that calculus / algebra /
physics / integral engines consume.

    parse("sin(x^2)")     -> ("sin", ("^", "x", 2))
    parse("2*x + 3 = 7")  -> ("=", ("+", ("*", 2, "x"), 3), 7)

Grammar (precedence low -> high):
    equation := expr ('=' expr)?
    expr     := term (('+' | '-') term)*
    term     := power (('*' | '/') power)*
    power    := unary ('^' power)?            # right-assoc
    unary    := '-' unary | atom
    atom     := number | func '(' expr ')' | ident | '(' expr ')'

Honest scope: explicit operators (2*x, not 2x), the functions sin/cos/exp/ln.
"""

import re

FUNCS = {"sin", "cos", "exp", "ln"}
_TOKEN = re.compile(r"\s*(\d+\.\d+|\d+|[A-Za-z_]\w*|\*\*|[+\-*/^()=])")


class ParseError(ValueError):
    pass


def _tokenize(s):
    toks, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(s, i)
        if not m:
            raise ParseError(f"unexpected character {s[i]!r}")
        tok = m.group(1)
        toks.append("^" if tok == "**" else tok)
        i = m.end()
    return toks


class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        node = self.expr()
        if self.peek() == "=":
            self.next()
            node = ("=", node, self.expr())
        if self.peek() is not None:
            raise ParseError(f"trailing tokens: {self.toks[self.i:]}")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.next()
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.power()
        while self.peek() in ("*", "/"):
            op = self.next()
            node = (op, node, self.power())
        return node

    def power(self):
        node = self.unary()
        if self.peek() == "^":
            self.next()
            node = ("^", node, self.power())          # right-assoc
        return node

    def unary(self):
        if self.peek() == "-":
            self.next()
            return ("neg", self.unary())
        return self.atom()

    def atom(self):
        t = self.next()
        if t is None:
            raise ParseError("unexpected end of expression")
        if t == "(":
            node = self.expr()
            if self.next() != ")":
                raise ParseError("missing ')'")
            return node
        if re.fullmatch(r"\d+\.\d+|\d+", t):
            return float(t) if "." in t else int(t)
        if re.fullmatch(r"[A-Za-z_]\w*", t):
            if t in FUNCS:
                if self.next() != "(":
                    raise ParseError(f"'{t}' needs parentheses")
                arg = self.expr()
                if self.next() != ")":
                    raise ParseError("missing ')'")
                return (t, arg)
            return t                                   # a variable / symbol
        raise ParseError(f"unexpected token {t!r}")


def parse(s):
    toks = _tokenize(s)
    if not toks:
        raise ParseError("empty expression")
    return _Parser(toks).parse()


def _demo():
    for s in ["x^3", "sin(x^2)", "2*x + 3 = 7", "x^2 + 3*x - 5",
              "exp(x) * ln(x)", "-cos(2*x)"]:
        print(f"  {s:18s} -> {parse(s)}")


if __name__ == "__main__":
    print("=== math_parser ===")
    _demo()
