#!/usr/bin/env python3
"""
distill_harness.py — resumable, rate-limit-aware dataset builder for distilling a
language front from a big teacher (e.g. deepseek-v3.1:671b-cloud via Ollama).

REVERSE distillation: we SEED the meaning (a Need / comparison) and ask the
teacher only for varied natural PHRASINGS of it. The label is the seed itself, so
the teacher cannot mislabel — it just supplies phrasing diversity (the one thing a
big model does that templates can't). Output trains a small LOCAL student later.

Built for a free tier: generate until the limit, back off, resume — losing
nothing. Every (question, label) is appended to JSONL immediately; on restart,
seeds already present are skipped. Hit the cap, rerun tomorrow, it continues.

    python3 distill_harness.py            # runs with a MOCK teacher (no network)
    # real use: pass OllamaClient('deepseek-v3.1:671b-cloud', host=<cloud>)

CHECK THE TEACHER'S ToS before using free-tier output to train a model.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

OUT = os.path.join(os.path.dirname(__file__), "distill_data.jsonl")


# ── seeds: the meanings we want phrasings for (the LABELS) ────────────────────
@dataclass(frozen=True)
class Seed:
    intent: str               # "attribute" | "compare"
    entity: str
    rel: str
    entity2: str = ""         # for compare
    comparative: str = ""     # for compare ("heavier")

    def sid(self):
        return f"{self.intent}:{self.entity}:{self.rel}:{self.entity2}:{self.comparative}"


def attribute_prompt(s: Seed, k):
    return (f"Give {k} different natural ways a person might ASK for the {s.rel} "
            f"of the {s.entity}. Vary phrasing, length, and directness. "
            f"One question per line, no numbering.")


def compare_prompt(s: Seed, k):
    return (f"Give {k} different natural ways to ASK whether the {s.entity} is "
            f"{s.comparative} than the {s.entity2} (comparing their {s.rel}). "
            f"One question per line, no numbering.")


# ── teacher clients ──────────────────────────────────────────────────────────
class RateLimit(Exception):
    pass


def _is_rate_limit(exc):
    m = str(exc).lower()
    return any(t in m for t in ("429", "rate limit", "too many", "quota", "limit reached"))


class MockTeacher:
    """Deterministic stand-in so the pipeline runs and tests without a network."""
    ATTR = ["what is the {rel} of the {entity}?",
            "how much {rel} does the {entity} have?",
            "tell me the {entity}'s {rel}",
            "find the {rel} for the {entity}",
            "{entity} {rel}?"]
    COMP = ["is the {entity} {comparative} than the {entity2}?",
            "which has more {rel}, the {entity} or the {entity2}?",
            "does the {entity} have more {rel} than the {entity2}?",
            "compare the {rel} of the {entity} and the {entity2}",
            "{entity} vs {entity2}: {comparative}?"]

    def generate(self, seed: Seed, k):
        tmpl = self.ATTR if seed.intent == "attribute" else self.COMP
        d = asdict(seed)
        return [t.format(**d) for t in tmpl[:k]]


class OllamaTeacher:                                # pragma: no cover (needs network)
    """Real teacher. host points at the Ollama cloud endpoint; model is e.g.
    'deepseek-v3.1:671b-cloud'. Raises RateLimit so the harness can back off."""
    def __init__(self, model="deepseek-v3.1:671b-cloud", host="http://localhost:11434"):
        self.model, self.host = model, host

    def generate(self, seed: Seed, k):
        import urllib.error
        import urllib.request
        prompt = attribute_prompt(seed, k) if seed.intent == "attribute" \
            else compare_prompt(seed, k)
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "think": False, "options": {"temperature": 0.9}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read()).get("response", "")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimit(str(e))
            raise
        except Exception as e:
            if _is_rate_limit(e):
                raise RateLimit(str(e))
            raise
        return _parse_lines(text)


def _parse_lines(text):
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*[-*\d.)\]]+\s*", "", ln).strip()   # strip bullets/numbers
        if ln and "?" in ln or (ln and len(ln.split()) >= 2):
            out.append(ln)
    return out


# ── resume bookkeeping ───────────────────────────────────────────────────────
def done_seeds(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["sid"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _label(seed: Seed):
    if seed.intent == "attribute":
        return {"entity": seed.entity, "rel": seed.rel}
    return {"x": seed.entity, "y": seed.entity2, "rel": seed.rel,
            "comparative": seed.comparative}


# ── the harness ──────────────────────────────────────────────────────────────
def build(seeds, teacher, path=OUT, k=8, max_backoff=6, base_sleep=10.0):
    done = done_seeds(path)
    todo = [s for s in seeds if s.sid() not in done]
    print(f"seeds: {len(seeds)} total, {len(done)} already done, {len(todo)} to do")
    written = 0
    with open(path, "a") as f:
        for s in todo:
            tries = 0
            while True:
                try:
                    qs = teacher.generate(s, k)
                    break
                except RateLimit:
                    tries += 1
                    if tries > max_backoff:
                        print("  rate limit persists — stopping. Rerun later to resume.")
                        return written
                    wait = base_sleep * (2 ** (tries - 1))
                    print(f"  rate limited; backoff {wait:.0f}s (try {tries})…")
                    time.sleep(wait)
            for q in dict.fromkeys(qs):             # dedup within the batch
                f.write(json.dumps({"question": q, "label": _label(s),
                                    "intent": s.intent, "sid": s.sid()}) + "\n")
                written += 1
            f.flush()                               # durable as we go
    print(f"wrote {written} examples -> {path}")
    return written


def _demo():
    seeds = [
        Seed("attribute", "rocket", "speed"),
        Seed("attribute", "rocket", "force"),
        Seed("attribute", "sample", "density"),
        Seed("compare", "rocket", "mass", "probe", "heavier"),
        Seed("compare", "rocket", "speed", "probe", "faster"),
    ]
    print("=== distill_harness — reverse distillation (MOCK teacher) ===\n")
    # first run
    build(seeds, MockTeacher(), k=5)
    # second run shows RESUME: everything already done, nothing re-generated
    print("\n-- rerun (simulating 'hit limit, came back later') --")
    build(seeds, MockTeacher(), k=5)

    # show a few rows
    print("\nsample of the dataset:")
    with open(OUT) as f:
        for line in list(f)[:4]:
            r = json.loads(line)
            print(f"  {r['question']:45s} -> {r['label']}")


if __name__ == "__main__":
    _demo()
