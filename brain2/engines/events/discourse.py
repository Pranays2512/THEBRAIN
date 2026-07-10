#!/usr/bin/env python3
"""discourse.py — the jump from sentence to paragraph.

Three cheap, crisp mechanisms; no learning yet — markers-first, like the earliest templates:

  * Coref     — a pronoun resolves to the most recent TYPE-COMPATIBLE entity on the context
                stack. Pure pointer resolution over working memory: symbolic and cheap.
  * Connectives — because/but/so/then become typed Relations (CAUSE/CONTRAST/SEQUENCE)
                between the surrounding event ids. Explicit markers only; implicit discourse
                stays unjudged (abstain, not guess).
  * ContextStack — the entities/events seen so far, persisting across sentences/turns = the
                dialogue state whole_brain wires in.

(Open-language track, Gap 3.)"""

from engines.events.event_form import Relation, CAUSE, CONTRAST, SEQUENCE

# marker -> (relation kind, direction). direction "fwd": Rel(prev -> cur); "bwd": Rel(cur -> prev)
# "because A B" reads cause after marker, so "A because B" => B causes A: cur causes prev.
_CONNECTIVES = {
    "because": (CAUSE, "bwd"),      # A because B  -> B CAUSE A
    "so":      (CAUSE, "fwd"),      # A so B       -> A CAUSE B
    "therefore": (CAUSE, "fwd"),
    "but":     (CONTRAST, "fwd"),
    "however": (CONTRAST, "fwd"),
    "then":    (SEQUENCE, "fwd"),
    "after":   (SEQUENCE, "fwd"),
}

_PRONOUNS = {"it", "they", "he", "she", "him", "her", "them", "its", "their"}


class ContextStack:
    """Most-recent-first memory of entities (with type) and event ids seen this discourse."""

    def __init__(self, type_of=None):
        self.entities = []                      # [(token, type)], most recent last
        self.events = []                        # event ids, in order
        self.type_of = type_of or (lambda _t: None)

    def push_entity(self, token):
        self.entities.append((token, self.type_of(token)))

    def push_event(self, eid):
        self.events.append(eid)

    def resolve(self, pronoun, want_type=None):
        """Most recent entity whose type is compatible with want_type. want_type None ->
        most recent entity of any type. Returns the token, or None if nothing compatible."""
        if pronoun not in _PRONOUNS:
            return None
        for tok, typ in reversed(self.entities):
            if want_type is None or typ is None or typ == want_type:
                return tok
        return None


def connective_of(token):
    """(kind, direction) if the token is a known discourse connective, else None."""
    return _CONNECTIVES.get(token)


def link_events(tokens, prev_eid, cur_eid):
    """If any token in the span between two events is a connective, emit the typed Relation.
    prev_eid/cur_eid are the ids of the events on either side of the marker."""
    for tok in tokens:
        c = connective_of(tok)
        if c is None:
            continue
        kind, direction = c
        if direction == "fwd":
            return Relation(kind, prev_eid, cur_eid)
        return Relation(kind, cur_eid, prev_eid)
    return None
