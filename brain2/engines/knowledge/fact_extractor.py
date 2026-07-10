#!/usr/bin/env python3
"""
fact_extractor.py — learn by READING (the inverse of production).

The conversation engine turns facts into sentences; this turns sentences back
into facts. For controlled, well-formed text it parses (subject, relation,
object) triples with grammar patterns — no LLM — and resolves "it"/"they" to the
running subject across sentences (the same coreference the conversation loop
uses). Feed it a paragraph; it learns the facts; then the brain reasons over
them.

    fe = FactExtractor()
    fe.extract("An apple is a fruit. It is red. It grows on a tree.")
      -> [("apple","isa","fruit"), ("apple","is","red"), ("apple","grows_on","tree")]
    fe.teach_into(text, conversation_engine)   # read -> stored -> queryable

Pluggable: FactExtractor.extract() is the interface. An LLM-based extractor for
MESSY/open text implements the same method and drops into the same slot —
offline, heavier, noisier — without changing anything downstream. This is the
clean, lightweight, verifiable side; the LLM is the upgrade for open text.

Honest scope: controlled declarative sentences (X is a Y / X has Y / X verbs Y,
with simple coreference). Messy open prose is the LLM's job.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

ARTICLES = {"a", "an", "the"}
PRONOUNS = {"it", "they", "he", "she", "this", "that"}

# (regex, relation). Order matters: specific first. "%MID%" means the relation
# is the MIDDLE captured group ("X is the parent of Y" -> (X, parent, Y)).
PATTERNS = [
    (r"^(\w+) is (?:the |a |an )?(\w+) of (\w+)$", "%MID%"),
    (r"^(\w+) is an? (\w+)$", "isa"),
    (r"^(\w+) are an? (\w+)$", "isa"),
    (r"^(\w+) (?:is|are) (\w+)$", "is"),
    (r"^(\w+) (?:has|have) an? (\w+)$", "has"),
    (r"^(\w+) (?:has|have) (\w+)$", "has"),
    (r"^(\w+) (?:grows?|grow) on an? (\w+)$", "grows_on"),
    (r"^(\w+) (?:lives?|live) in an? (\w+)$", "lives_in"),
    (r"^(\w+) (?:gives?|give) (\w+)$", "gives"),
    (r"^(\w+) (?:eats?|eat) an? (\w+)$", "eats"),
    (r"^(\w+) (\w+) an? (\w+)$", None),       # generic: X VERB a Y
    (r"^(\w+) (\w+) (\w+)$", None),           # generic: X VERB Y
]


def _norm(word):
    return word.lower().strip()


def _stem(verb):
    v = verb.lower()
    if v.endswith("es"):
        return v[:-2]
    if v.endswith("s"):
        return v[:-1]
    return v


class FactExtractor:
    """Grammar-based extractor for controlled text. Subclass and override
    extract() for an LLM-based extractor over messy text."""

    @staticmethod
    def _sentences(text):
        return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]

    @staticmethod
    def _clean(sent):
        # lowercase, drop leading article, drop punctuation/commas
        words = re.findall(r"[a-zA-Z']+", sent.lower())
        if words and words[0] in ARTICLES:
            words = words[1:]
        # drop an article right after the subject's verb is handled by patterns;
        # remove stray articles inside the middle for the generic case
        return words

    def _extract_one(self, sent, last_subj):
        words = self._clean(sent)
        if not words:
            return None, last_subj
        # coreference: pronoun subject -> the running subject
        if words[0] in PRONOUNS:
            if last_subj is None:
                return None, last_subj
            words[0] = last_subj
        joined = " ".join(words)
        for pat, rel in PATTERNS:
            m = re.match(pat, joined)
            if not m:
                continue
            if rel == "%MID%":                   # "X is the R of Y" -> (X, R, Y)
                subj, rel, obj = m.group(1), _norm(m.group(2)), m.group(3)
            elif rel is None:                    # generic: relation is the verb
                subj, verb, obj = m.group(1), m.group(2), m.group(3)
                if verb in ARTICLES or obj in ARTICLES:
                    continue
                rel = _stem(verb)
            else:
                subj, obj = m.group(1), m.group(2)
            subj, obj = _norm(subj), _norm(obj)
            if subj in ARTICLES or obj in ARTICLES or subj == obj:
                continue
            return (subj, rel, obj), subj
        return None, last_subj

    def extract(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        triples, last_subj = [], None
        for sent in self._sentences(text):
            triple, last_subj = self._extract_one(sent, last_subj)
            if triple:
                triples.append(triple)
        return triples

    def teach_into(self, text, engine):
        """Extract facts and learn them into a ConversationEngine /
        KnowledgeEngine (anything with .learn(s, r, o)). Returns count learned."""
        n = 0
        for s, r, o in self.extract(text):
            if engine.learn(s, r, o):
                n += 1
        return n


def _demo():
    from faculties.conversation_engine import ConversationEngine
    fe = FactExtractor()
    text = ("An apple is a fruit. It is red. It grows on a tree. It has seeds. "
            "A dog is an animal. It has a tail.")
    print("Reading text into facts:")
    for t in fe.extract(text):
        print(f"    {t}")
    c = ConversationEngine()
    learned = fe.teach_into(text, c)
    print(f"\n  learned {learned} facts; now answering from what it READ:")
    for q in ["what is apple?", "is it red?", "what is dog?"]:
        print(f"  > {q}\n    {c.respond(q)}")


if __name__ == "__main__":
    _demo()
