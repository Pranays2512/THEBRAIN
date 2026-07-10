#!/usr/bin/env python3
"""event_parse.py — the INTAKE the event membrane was waiting for.

Until now templates only emitted stative FACTs (unconstrained verbs), so event_verify's real
work — constrained verbs, negation, tense, causality — never fired on prose. This turns a
sentence into an Event(verb, agent, patient, time, polarity):

  * negation  — not / never / -n't  -> polarity NEG (the claim is denied)
  * tense     — did/was/-ed/irregular-past -> past; will/shall -> future; else present
  * SVO       — agent = nearest entity/pronoun before the verb, patient = nearest after

Markers-first and crisp (same stage as discourse.py) — no learning yet; a bad parse yields
None (abstain), it never guesses an Event into existence. Pronouns stay as tokens for the
reader's coref to resolve. verbs is the known-lemma set (crisp verb identification).
(Open-language track — closes the intake gap: now prose reaches the membrane.)"""

import re

from core.store.parse_template import normalize
from event_form import Event, POS, NEG
from discourse import _PRONOUNS

_CONTRACTIONS = {
    "didn't": "did not", "don't": "do not", "doesn't": "does not", "won't": "will not",
    "can't": "can not", "cannot": "can not", "isn't": "is not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not", "aren't": "are not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not", "never": "not",
}
_NEG = {"not", "no"}
_PAST_AUX = {"did", "was", "were", "had"}
_FUTURE_AUX = {"will", "shall"}
# irregular pasts + a few regulars whose stemming wouldn't recover the lemma
_IRREGULAR = {"ate": "eat", "ran": "run", "went": "go", "saw": "see", "made": "make",
              "drank": "drink", "flew": "fly", "caught": "catch", "chased": "chase",
              "liked": "like", "moved": "move", "was": "be", "were": "be"}


def _lemma(tok):
    return _IRREGULAR.get(tok) or normalize(tok)


# determiners / auxiliaries / negation / connectives — never a role filler on their own
_STOP = {"the", "a", "an", "this", "that", "these", "those", "did", "do", "does", "not", "no",
         "was", "were", "is", "are", "am", "be", "been", "will", "shall", "had", "has", "have",
         "to", "then", "so", "but", "because", "and", "of", "at"}


def _nearest(seq, entities, type_of):
    """The role filler nearest the verb. Prefer a real referent (known entity / pronoun / a
    token the type oracle knows); else fall back to the nearest CONTENT token so an UNKNOWN
    word still surfaces as the role — the membrane must get the chance to abstain on it, not
    silently drop it to None."""
    seq = list(seq)
    for t in seq:
        if t in entities or t in _PRONOUNS or (type_of and type_of(t)):
            return t
    for t in seq:
        if t not in _STOP:
            return t
    return None


def parse_event(sentence, entities, verbs, type_of=None):
    """Sentence -> Event, or None if no known verb is found (abstain, never guess)."""
    text = sentence.lower()
    for c, e in _CONTRACTIONS.items():
        text = text.replace(c, e)
    raw = re.findall(r"[a-z_]+", text)

    polarity = NEG if any(t in _NEG for t in raw) else POS
    tense = ("past" if any(t in _PAST_AUX for t in raw)
             else "future" if any(t in _FUTURE_AUX for t in raw) else "present")

    vi = verb = None
    for i, t in enumerate(raw):                          # 1. trusted lemma (highest confidence)
        lem = _lemma(t)
        if lem in verbs or t in verbs:
            vi, verb = i, lem
            break
    if vi is None:                                       # 2. positional: verb = first content
        content = [i for i, t in enumerate(raw)          #    token after the subject. Recovers
                   if t not in _STOP and t not in _NEG]  #    STRUCTURE on an unknown verb — but
        if len(content) < 2:                             #    the caller must ABSTAIN on it (the
            return None                                  #    verb isn't trusted), never admit.
        vi, verb = content[1], _lemma(raw[content[1]])
    if tense == "present" and (raw[vi] in _IRREGULAR or raw[vi].endswith("ed")):
        tense = "past"                                  # verb form itself carries the tense

    agent = _nearest(reversed(raw[:vi]), entities, type_of)
    patient = _nearest(raw[vi + 1:], entities, type_of)
    return Event(verb, agent, patient, tense, polarity)


def verb_trusted(ev, verbs):
    """True if the verb came from the trusted lexicon (event is admissible); False if it was
    only recovered positionally (structure parsed, but the verb is unverifiable -> abstain).
    This is what keeps positional coverage from flooding the truth store with wild events."""
    return ev is not None and ev.verb in verbs
