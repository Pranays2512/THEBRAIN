#!/usr/bin/env python3
"""
appraisal_engine.py — the input's pragmatic frame (and the redesigned emotion).

Before understanding WHAT an utterance means, recognize what KIND it is: a
question? a greeting? a command? and its tone — curious? friendly? Humans do
this from FORM ("what", inversion "are you", "?") before meaning. This grades
each word along pragmatic/affect dimensions, surprise-weighted so high-frequency
"constant" words ("are you") barely count and the informative markers carry the
signal.

This is emotion's real job — a fast, low-dimensional APPRAISAL of the input —
not the weak learning-rate modulator it was. It does NOT understand content
(that's the wall); it carves off the tractable pragmatic slice in front of it.

    AppraisalEngine().appraise("hey, how are you?")
      -> frame {greeting, friendly, question, curious, about_self}, type 'question'
"""

import re

# word -> {dimension: weight}. Informative markers; everything else is content.
MARKERS = {
    "what": {"question": 1, "definition": 1},
    "which": {"question": 1, "definition": 1},
    "how": {"question": 1, "curious": 1},
    "why": {"question": 1, "curious": 1},
    "who": {"question": 1}, "where": {"question": 1},
    "when": {"question": 1}, "whose": {"question": 1},
    "?": {"question": 1},
    "do": {"question": 0.5}, "does": {"question": 0.5}, "did": {"question": 0.5},
    "is": {"question": 0.3}, "are": {"question": 0.3}, "can": {"question": 0.5},
    "could": {"question": 0.5, "polite": 1}, "will": {"question": 0.4},
    "would": {"question": 0.4, "polite": 1},
    "hi": {"greeting": 1, "friendly": 1}, "hey": {"greeting": 1, "friendly": 1},
    "hello": {"greeting": 1, "friendly": 1}, "yo": {"greeting": 1, "friendly": 1},
    "thanks": {"friendly": 1}, "thank": {"friendly": 1}, "please": {"polite": 1},
    "you": {"about_self": 1}, "your": {"about_self": 1}, "yourself": {"about_self": 1},
    "tell": {"command": 1}, "show": {"command": 1}, "give": {"command": 1},
    "list": {"command": 1}, "describe": {"command": 1}, "explain": {"command": 1},
    "not": {"negation": 1}, "no": {"negation": 1}, "never": {"negation": 1},
}

# very common function words: low information ("constant patterns"), down-weighted
COMMON = {"a", "an", "the", "of", "to", "i", "it", "and", "that", "this", "in",
          "on", "for", "with", "me", "my", "be", "am", "was", "were", "of"}

DIMENSIONS = ["question", "greeting", "command", "curious", "friendly",
              "polite", "about_self", "definition", "negation"]


class Appraisal:
    def __init__(self, frame, utterance_type):
        self.frame = frame              # dict dimension -> score
        self.type = utterance_type      # 'question' | 'greeting' | 'command' | 'statement'

    def __repr__(self):
        active = {k: round(v, 2) for k, v in self.frame.items() if v > 0}
        return f"Appraisal(type={self.type!r}, {active})"


class AppraisalEngine:
    def __init__(self):
        self.frame_dims = DIMENSIONS

    @staticmethod
    def _tokens(text):
        # words + a '?' token if present
        toks = re.findall(r"[a-zA-Z']+", text.lower())
        if "?" in text:
            toks.append("?")
        return toks

    def _weight(self, token):
        # surprise weighting: common function words carry little signal
        return 0.3 if token in COMMON else 1.0

    def appraise(self, text):
        if not isinstance(text, str) or not text.strip():
            return Appraisal({d: 0.0 for d in DIMENSIONS}, "statement")
        frame = {d: 0.0 for d in DIMENSIONS}
        for tok in self._tokens(text):
            marks = MARKERS.get(tok)
            if not marks:
                continue
            w = self._weight(tok)
            for dim, val in marks.items():
                frame[dim] += val * w
        return Appraisal(frame, self._classify(frame))

    @staticmethod
    def _classify(frame):
        # question dominates if present; then greeting/command; else statement
        if frame["question"] >= 0.8:
            return "question"
        if frame["greeting"] >= 1.0 and frame["question"] < 0.8:
            return "greeting"
        if frame["command"] >= 1.0:
            return "command"
        if frame["question"] >= 0.4:        # weak inversion-only question
            return "question"
        return "statement"


def _demo():
    ae = AppraisalEngine()
    for t in ["hey, how are you?", "what is apple?", "do you know him?",
              "tell me about you", "the apple is red"]:
        print(f"  {t!r:28} -> {ae.appraise(t)}")


if __name__ == "__main__":
    _demo()
