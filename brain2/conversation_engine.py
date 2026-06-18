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
from reasoning_engine import ReasoningEngine
from appraisal_engine import AppraisalEngine

PRONOUNS = {"it", "that", "this", "he", "she", "him", "her", "they", "them"}
# words that are never the *entity* of a query
NON_ENTITY = {
    "what", "which", "how", "why", "who", "where", "when", "whose",
    "is", "are", "am", "do", "does", "did", "can", "could", "will", "would",
    "a", "an", "the", "of", "to", "i", "me", "my", "you", "your", "it", "that",
    "this", "in", "on", "for", "with", "and", "tell", "show", "describe",
    "explain", "give", "list", "about", "know", "please", "hey", "hi", "hello",
    "yes", "no", "not", "him", "her", "them", "they", "he", "she",
}

# relation -> sentence template (controlled vocabulary)
TEMPLATES = {
    "isa":      "is {art} {obj}",
    "is":       "is {art} {obj}",
    "color":    "is {obj}",
    "grows_on": "grows on {art} {obj}",
    "has":      "has {obj}",
    "gives":    "gives {obj}",
    "lives_in": "lives in {art} {obj}",
}


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def verb_be(subject):
    return "are" if subject.endswith("s") and not subject.endswith("ss") else "is"


class ConversationEngine:
    def __init__(self):
        self.r = ReasoningEngine()
        self.appraiser = AppraisalEngine()
        self.topic = None                      # working memory: entity in focus

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

    def _confirm(self, subj, target):
        objs = {o for (_, o) in self._facts_of(subj)}
        if target in objs:
            return f"Yes, {self._sentence(subj, *next((r, o) for (r, o) in self._facts_of(subj) if o == target))}."
        # maybe target is a category reached transitively / by rule
        for rel in {r for (r, _) in self._facts_of(subj)}:
            ans, _ = self.r.ask(subj, rel)
            if ans == target:
                return f"Yes, {subj} {rel} {target}."
        return f"Not that I know of."

    def _sentence(self, subj, rel, obj):
        tmpl = TEMPLATES.get(rel, f"{rel.replace('_', ' ')} {{obj}}")
        body = tmpl.format(art=article(obj), obj=obj).replace("is ", verb_be(subj) + " ", 1) \
            if tmpl.startswith("is") else tmpl.format(art=article(obj), obj=obj)
        return f"{subj} {body}"

    def _describe(self, subj):
        facts = self._facts_of(subj)
        if not facts:
            return f"I don't know anything about {subj}."
        sentences = []
        for i, (rel, obj) in enumerate(facts):
            s = self._sentence(subj if i == 0 else "it", rel, obj)
            sentences.append(s[0].upper() + s[1:] + ".")
        # first sentence: add an article to the subject if it's a singular noun
        first = sentences[0]
        if not subject_is_proper(subj):
            sentences[0] = f"{article(subj).capitalize()} {first[0].lower() + first[1:]}"
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
