#!/usr/bin/env python3
"""mouth.py — the OWNED generative mouth: structure -> English, learned like a child.

Not an LLM. The insight: a learned Template maps tokens <-> slots, so the SAME grammar the
brain acquired by comprehension runs BACKWARD to produce. The mouth speaks with the
constructions it learned to parse, and gets more fluent as it reads more — exactly child
language acquisition (comprehension grammar = production grammar).

Membrane: the mouth only renders VERIFIED structure (an Event the membrane admitted, a fact
the store owns). It never invents content — grammar dresses truth. Fluency is child-grade
early (imperfect morphology, like a kid saying 'goed'); it improves with learned templates and
morphology, and a tuned LLM stays an anytime fluency fallback.

Two surfaces:
  * say_event(Event)         — agent verb(s)/did-not-verb patient, tense + polarity realized
  * say_fact(e, rel, v, tm)  — bidirectional: fill a learned statement template's slots
"""

from core.events.event_form import NEG

# lemma -> past (generation direction; the inverse of event_parse's surface->lemma map)
_PAST = {"eat": "ate", "run": "ran", "go": "went", "see": "saw", "make": "made",
         "drink": "drank", "fly": "flew", "catch": "caught", "chase": "chased",
         "like": "liked", "move": "moved", "be": "was"}

# LEARNED morphology: lemma -> {"past":..., "present3sg":...}. Populated by the data loader
# (MORPH lines). When present it overrides the guessed rules -> the mouth graduates from
# child-grade ("the drone weigh") to fluent ("the drone weighs") by LEARNING, not by an LLM.
MORPH = {}


def _realize(verb, tense, polarity):
    """Surface form of a verb given tense + polarity. Prefers LEARNED morphology (MORPH), then
    the small irregular table, then regular rules. Imperfect on unseen irregulars — honest."""
    m = MORPH.get(verb, {})
    if polarity == NEG:
        aux = "did not" if tense == "past" else "does not"
        return "%s %s" % (aux, verb)                  # negation uses the base form
    if tense == "past":
        return m.get("past") or _PAST.get(verb) or (verb + "d" if verb.endswith("e") else verb + "ed")
    if tense == "future":
        return "will " + verb
    if m.get("present3sg"):
        return m["present3sg"]
    # present 3rd-singular (regular fallback)
    if verb.endswith(("s", "sh", "ch", "x", "z")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _det(noun):
    return "an" if noun[:1] in "aeiou" else "a"


def say_event(ev):
    """Render an Event to a sentence. Only the roles the event carries are spoken."""
    verb = _realize(ev.verb, ev.time or "present", ev.polarity)
    parts = ["the", ev.agent] if ev.agent else []
    parts.append(verb)
    if ev.patient:
        parts += ["the", ev.patient]
    return " ".join(parts).strip().capitalize() + "."


def ask(ev):
    """Turn an event into a curiosity QUESTION — the mouth's role when surprise drives the
    brain to wonder. 'the dog ate the fish' (surprising) -> 'Why did the dog eat the fish?'"""
    aux = "did" if (ev.time or "present") == "past" else "does"
    q = ["why", aux, "the", ev.agent or "it", ev.verb]
    if ev.patient:
        q += ["the", ev.patient]
    return " ".join(q).capitalize() + "?"


def say_fact(entity, rel, value, tm):
    """Render (entity, rel, value) by running a learned statement template BACKWARD: fill its
    entity/value slots, emit its literal words in order. Returns None if the grammar hasn't
    learned a construction for this relation yet (honest silence, not a guess)."""
    for t in tm.templates:
        if t.rel != rel or not any(it[0] == "slot" and it[1] == "value" for it in t.items):
            continue
        out = []
        for it in t.items:
            if it[0] == "w":
                out.append(it[1])
            elif it[0] == "any":
                out.append("a")
            elif it[1] == "entity":
                out.append(str(entity))
            elif it[1] == "value":
                out.append(_fmt(value))
        return " ".join(out).capitalize() + "."
    return None


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
