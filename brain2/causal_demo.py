#!/usr/bin/env python3
"""
causal_demo.py — "how / why" questions: the brain narrates a CAUSAL CHAIN.

A process is a chain on one causal relation (leads_to, helps): each step causes
the next. Taught the steps, the brain answers "how does X happen?" by walking
that chain and narrating it as a sentence — backward to X's causes for
"how does X grow/form", forward to X's effects for "how does X help".

This reuses the reasoning core (transitive traversal) — a "how" question is just
a chain walk with a direction. No new reasoning power, a new INTENT + narrator.

Honest scope: linear taught processes, controlled phrasing. It does not discover
the science; it reorders and verbalizes the steps it was given.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from conversation_engine import ConversationEngine


def main():
    print("=== causal_demo — how / why, answered as a chain ===\n")
    c = ConversationEngine()

    # taught process 1: how an apple grows (a causal chain)
    for s, o in [("sunlight", "photosynthesis"), ("photosynthesis", "sugar"),
                 ("sugar", "fruit"), ("fruit", "apple")]:
        c.learn(s, "leads_to", o)
    # taught process 2: how a vitamin helps the body
    for s, o in [("vitamin", "immune_system"), ("immune_system", "fighting_infection"),
                 ("fighting_infection", "good_health")]:
        c.learn(s, "helps", o)
    c.set_transitive("leads_to")
    c.set_transitive("helps")

    for q in [
        "how does an apple grow?",          # backward: causes of apple
        "why does an apple grow?",          # same chain, 'why'
        "how does vitamin help the body?",  # forward: effects of vitamin
        "how does a rock grow?",            # honest unknown
    ]:
        print(f"  > {q}")
        print(f"    {c.respond(q)}\n")


if __name__ == "__main__":
    main()
