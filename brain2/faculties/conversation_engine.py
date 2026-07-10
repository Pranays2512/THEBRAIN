#!/usr/bin/env python3
"""
conversation_engine.py — the understand -> reason -> produce loop (capstone).

Ties the whole stack together for CONTROLLED conversation, fully symbolic and
explainable:

  understand : AppraisalEngine (utterance type/tone) + intent recognition +
               working-memory context (resolves "it" / "that" to the topic)
  reason     : ReasoningEngine (facts, rules, transitive) — the hardened core
  produce    : grammar-based verbalization of the retrieved relations
               (articles a/an, is/are agreement) — generated, not pattern-matched

    c = ConversationEngine()
    c.learn("apple", "isa", "fruit"); c.learn("apple", "color", "red")
    c.respond("what is apple?")  -> "An apple is a fruit. It is red."
    c.respond("is it red?")      -> "Yes."        ("it" -> apple, from context)

Honest scope: controlled conversation. Intent recognition is form-based over a
defined set of question shapes; genuine open-domain comprehension is the wall
(that needs an LLM). Within the controlled set, every word out is derived from a
stored relation through a grammar rule.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.reasoning.reasoning_engine import ReasoningEngine
from faculties.appraisal_engine import AppraisalEngine
from core.math.word_math import solve as solve_word_math

PRONOUNS = {"it", "that", "this", "he", "she", "him", "her", "they", "them"}
# "how does X grow/form?" asks what PRODUCES X -> walk the causal chain backward;
# anything else ("how does X help/work?") walks forward to its effects.
BACKWARD_WORDS = {"grow", "grows", "grew", "form", "forms", "happen", "happens",
                  "made", "produced", "created", "develop", "develops", "appear"}
# words that are never the *entity* of a query
NON_ENTITY = {
    "what", "which", "how", "why", "who", "where", "when", "whose", "many", "much",
    "is", "are", "am", "do", "does", "did", "can", "could", "will", "would",
    "a", "an", "the", "of", "to", "i", "me", "my", "you", "your", "it", "that",
    "this", "in", "on", "for", "with", "and", "tell", "show", "describe",
    "explain", "give", "list", "about", "know", "please", "hey", "hi", "hello",
    "yes", "no", "not", "him", "her", "them", "they", "he", "she",
}

# relation -> sentence template (controlled vocabulary)
TEMPLATES = {
    "isa":      "is {art} {obj}",
    "is":       "is {obj}",            # adjective/property: "is red" (no article)
    "color":    "is {obj}",
    "grows_on": "grows on {art} {obj}",
    "has":      "has {obj}",
    "gives":    "gives {obj}",
    "lives_in": "lives in {art} {obj}",
    # ConceptNet common-sense relations
    "can":      "can {obj}",
    "used_for": "is used for {obj}",
    "part_of":  "is part of {art} {obj}",
    "made_of":  "is made of {obj}",
}


# relation -> (clause verb/prefix, objects take an article) for AGGREGATION:
# many objects under one relation collapse into a single coordinated clause
# ("is a canine, a pet and an animal") instead of one sentence each.
CLAUSES = {
    "isa": ("is", True), "is": ("is", False), "color": ("is", False),
    "has": ("has", False), "gives": ("gives", False),
    "grows_on": ("grows on", True), "lives_in": ("lives in", True),
    "can": ("can", False), "used_for": ("is used for", False),
    "part_of": ("is part of", True), "made_of": ("is made of", False),
}


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def oxford(items):
    """['a', 'b', 'c'] -> 'a, b and c'  (single item passes through)."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def verb_be(subject):
    return "are" if subject.endswith("s") and not subject.endswith("ss") else "is"


class ConversationEngine:
    def __init__(self, max_describe=None):
        self.r = ReasoningEngine()
        self.appraiser = AppraisalEngine()
        self.topic = None                      # working memory: entity in focus
        self.max_describe = max_describe       # cap facts per describe (dense KBs)

    # ── teaching ─────────────────────────────────────────────────────────────
    def learn(self, subj, rel, obj):
        return self.r.learn(subj, rel, obj)

    def add_rule(self, a, b, c):
        self.r.add_rule(a, b, c)

    def set_transitive(self, rel):
        self.r.set_transitive(rel)

    # ── understanding helpers ────────────────────────────────────────────────
    def _entities(self, text):
        return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in NON_ENTITY]

    def _has_pronoun(self, text):
        return any(w in PRONOUNS for w in re.findall(r"[a-z']+", text.lower()))

    # ── the loop ─────────────────────────────────────────────────────────────
    def respond(self, text):
        ap = self.appraiser.appraise(text)
        if ap.type == "greeting":
            return "Hello! Ask me about something you've taught me."

        ents = self._entities(text)
        # coreference: a pronoun (or nothing) -> the current topic
        if not ents and self.topic:
            ents = [self.topic]
        elif self._has_pronoun(text) and self.topic and self.topic not in ents:
            ents = [self.topic] + ents

        if not ents:
            return "I'm not sure what you're asking about."

        subject = ents[0]
        self.topic = subject                   # update working-memory focus

        raw = re.findall(r"[a-z']+", text.lower())

        # arithmetic word problem: "... how many ... left/have?"
        if "how" in raw and "many" in raw:
            ans = solve_word_math(text)
            if ans:
                return ans

        # how/why/ways-question: narrate the causal chain(s) through the topic
        if {"how", "why", "way", "ways"} & set(raw):
            return self._how(ents, raw)

        # confirm-question: "is X (a) Y" / "is it red" -> check a value
        if len(ents) >= 2 or (self._has_pronoun(text) and len(ents) >= 1 and ents[0] != subject):
            target = ents[-1]
            return self._confirm(subject, target)
        if ap.frame.get("about_self") and subject in NON_ENTITY:
            return "I am a small reasoning engine that learns facts and answers about them."

        # describe / lookup
        return self._describe(subject)

    # ── reason + produce ─────────────────────────────────────────────────────
    def _facts_of(self, subj):
        return [(r, o) for (s, r, o) in self.r.kb.facts if s == subj]

    def _how(self, ents, raw):
        """Answer 'how/why does X ...?' by narrating the longest causal chain
        through X over a transitive (causal) relation — backward to its causes
        for 'how does X form/grow', forward to its effects otherwise."""
        subjects = {s for s, _, _ in self.r.kb.facts}
        topic = next((e for e in ents if e in subjects), ents[0])
        self.topic = topic
        direction = "backward" if (set(raw) & BACKWARD_WORDS) else "forward"

        # "in which ways …?" — list every forward branch, not one chain
        if "way" in raw or "ways" in raw:
            ways = self._how_ways(topic)
            if ways:
                return ways

        # pick the relation giving the longest chain; try the other way if none
        best, best_rel = [], None
        for d in (direction, "forward" if direction == "backward" else "backward"):
            for rel in self.r.transitive:
                chain = self.r.process_chain(topic, rel, d)
                if len(chain) > len(best):
                    best, best_rel = chain, rel
            if len(best) >= 2:
                break
        if len(best) < 2:
            return f"I don't know how {topic.replace('_', ' ')} works."
        return self._narrate_chain(best, best_rel)

    def _how_ways(self, topic):
        """List every forward branch from the topic — 'in which ways does X
        help?' -> one clause per branch, later ones de-duplicated to 'It also'."""
        best, best_rel = [], None
        for rel in self.r.transitive:
            branches = self.r.process_branches(topic, rel)
            if len(branches) > len(best):
                best, best_rel = branches, rel
        if len(best) < 2:                          # 0 or 1 branch -> not a list
            return None
        sents = []
        for i, branch in enumerate(best):
            s = self._narrate_chain(branch, best_rel)
            if i > 0:                              # drop the repeated subject
                s = "It also " + s.split(" ", 1)[1]
            sents.append(s)
        return " ".join(sents)

    def _narrate_chain(self, path, rel):
        verb = rel.replace("_", " ")
        parts = [p.replace("_", " ") for p in path]
        sent = parts[0]
        for i, nxt in enumerate(parts[1:]):
            sep = f" {verb} " if i == 0 else f", which {verb} "
            sent += f"{sep}{nxt}"
        return sent[0].upper() + sent[1:] + "."

    def _confirm(self, subj, target):
        objs = {o for (_, o) in self._facts_of(subj)}
        if target in objs:
            return f"Yes, {self._sentence(subj, *next((r, o) for (r, o) in self._facts_of(subj) if o == target))}."
        # category membership reached transitively — show the chain
        # (dog -> pet -> animal), via the engine's multi-parent closure
        for rel in self.r.transitive:
            ok, path = self.r.reaches(subj, rel, target)
            if ok:
                return f"Yes — {' -> '.join(path)}."
        # derivable by any relation or composition rule (all bindings)
        for rel in {r for (r, _) in self._facts_of(subj)} | set(self.r.transitive):
            if target in self.r.ask_all(subj, rel):
                return f"Yes, {subj} {rel} {target}."
        return f"Not that I know of."

    def _sentence(self, subj, rel, obj):
        disp = obj.replace("_", " ")                 # motor_vehicle -> "motor vehicle"
        tmpl = TEMPLATES.get(rel, f"{rel.replace('_', ' ')} {{obj}}")
        body = tmpl.format(art=article(disp), obj=disp).replace("is ", verb_be(subj) + " ", 1) \
            if tmpl.startswith("is") else tmpl.format(art=article(disp), obj=disp)
        return f"{subj.replace('_', ' ')} {body}"

    def _clause(self, head, rel, objs):
        """One coordinated clause for ALL objects of a relation:
        ("dog","isa",[canine,pet,animal]) -> "is a canine, a pet and an animal"."""
        prefix, takes_art = CLAUSES.get(rel, (rel.replace("_", " "), False))
        if prefix == "is":
            prefix = verb_be(head)                   # is/are agreement
        phrases = []
        for o in objs:
            d = o.replace("_", " ")
            phrases.append(f"{article(d)} {d}" if takes_art else d)
        return f"{prefix} {oxford(phrases)}"

    def _describe(self, subj):
        facts = self._facts_of(subj)
        if not facts:
            return f"I don't know anything about {subj}."
        # group objects under each relation (aggregation), leading with isa
        groups = {}
        for rel, obj in facts:
            objs = groups.setdefault(rel, [])
            if obj not in objs:
                objs.append(obj)
        ordered = sorted(groups.items(), key=lambda kv: 0 if kv[0] == "isa" else 1)
        if self.max_describe:                        # cap clauses AND objects/clause
            ordered = [(r, objs[:self.max_describe]) for r, objs in ordered[:self.max_describe]]

        sentences = []
        for i, (rel, objs) in enumerate(ordered):
            head = subj if i == 0 else "it"          # pronoun after the first clause
            s = f"{head.replace('_', ' ')} {self._clause(head, rel, objs)}"
            sentences.append(s[0].upper() + s[1:] + ".")
        # first sentence: add an article to the subject if it's a singular noun
        if not subject_is_proper(subj):
            first = sentences[0]
            sentences[0] = f"{article(subj.replace('_', ' ')).capitalize()} {first[0].lower() + first[1:]}"
        return " ".join(sentences)


def subject_is_proper(word):
    # heuristic: treat capitalized-looking / name-like as proper (no article)
    return False                                # controlled demo: common nouns


def _demo():
    c = ConversationEngine()
    for s, r, o in [("apple", "isa", "fruit"), ("apple", "color", "red"),
                    ("apple", "grows_on", "tree"), ("apple", "has", "seeds")]:
        c.learn(s, r, o)
    print("ConversationEngine demo:")
    for q in ["hello", "what is apple?", "is apple a fruit?", "is it red?",
              "is it blue?", "what is banana?"]:
        print(f"  > {q}")
        print(f"    {c.respond(q)}")


if __name__ == "__main__":
    _demo()
