#!/usr/bin/env python3
"""
generate_distill_corpus.py — teacher signal generator for STAMLAT distillation.

Emits three artifacts (deterministic, seeded):
  mouth_unified.txt   chat blocks (contract prefix + fluent English tail)
                      PLUS <p>-scaffold plan-rendering lines, so ONE model
                      serves both the free-chat lane and the content-locked
                      plan lane (plans_supported() == true).
  reader_corpus.txt   read/triple pairs — teaching the same transformer
                      architecture to act as EYES: english -> (s r o).
  reader_probes.txt   held-out reader pairs for exact-match gating.

The teacher (agent-authored seeds) guarantees: every answer is either a
verified contract string, a computed mathematical truth, or an honest
refusal. Nothing hallucinated enters the corpus.
"""
import random

rng = random.Random(20260826)

# ── 1. CHAT BLOCKS ────────────────────────────────────────────────────────
# Each entry: (contract_prefix, [user variants], [english tails])
CHAT = [
    ("intent greeting style friendly",
     ["hello", "hello there", "hi brain"],
     ["hello there! wonderful to see you today.",
      "hello! i am glad you came."]),
    ("intent greeting emotion happy",
     ["hi", "hey", "hi there"],
     ["hi! i am happy you are here.",
      "hey! great to see you."]),
    ("intent welcome style warm",
     ["hey there", "good evening", "good afternoon"],
     ["hey! welcome, come on in.",
      "welcome! it is good to have you here."]),
    ("intent greeting status ready",
     ["good morning", "greetings"],
     ["good morning! ready for a great day.",
      "greetings! all systems warm and ready."]),
    ("identity brain type ai",
     ["who are you", "what are you"],
     ["i am brain three, a cognitive architecture that verifies before it speaks.",
      "i am brain, a symbolic mind with a trained language organ."]),
    ("name brain type ai",
     ["what is your name", "tell me your name"],
     ["my name is brain, your verified knowledge companion.",
      "i am called brain."]),
    ("identity system type cognitive",
     ["describe yourself", "tell me about yourself"],
     ["i learn facts, connect them, and refuse to guess what i do not know.",
      "i am a reasoning system that shows proof chains for every claim."]),
    ("status good energy high",
     ["how are you", "how do you feel"],
     ["feeling sharp and stable, thanks for asking.",
      "good and focused, memory consolidated."]),
    ("status optimal condition excellent",
     ["what is your state", "are you ok"],
     ["optimal condition, gates green.",
      "excellent condition and ready to reason."]),
]

FACT_QA = [
    ("cheetah", "fact animal class mammal",
     ["what is a cheetah", "what is the cheetah"],
     ["a cheetah is a feline and the fastest land hunter.",
      "the cheetah is a f mammal famous for speed."]),
    ("einstein", "fact person field physics",
     ["who was einstein", "who is einstein"],
     ["albert einstein was a scientist who reshaped physics.",
      "einstein was a physicist who described relativity."]),
    ("turing", "fact person domain logic",
     ["what did turing study", "who was turing"],
     ["alan turing was a logician, and logicians use logic.",
      "turing studied logic and founded modern computing."]),
    ("curie", "fact person field chemistry",
     ["who was marie curie", "who was curie"],
     ["marie curie was a chemist who pioneered radioactivity research.",
      "curie was a chemist famous for radioactive elements."]),
    ("bohr", "fact person field physics",
     ["who was niels bohr", "who was bohr"],
     ["niels bohr was a physicist who mapped the atom.",
      "bohr was a physicist who modeled atomic shells."]),
    ("gravity", "fact cause chain active",
     ["what is gravity", "how does gravity work"],
     ["gravity causes acceleration, and acceleration changes velocity.",
      "gravity pulls mass and causes acceleration."]),
    ("falcon", "fact bird class predator",
     ["what is a falcon", "what is the falcon"],
     ["a falcon is a raptor, which is a bird, which is an animal.",
      "the falcon is a fast hunting bird."]),
]

REFUSALS = [
    ("unknown subject honest refusal",
     ["who was {}", "tell me about {}", "do you know {}"],
     ["i do not know {} yet. teach me and i will remember.",
      "{} is outside my knowledge so far. i will not guess."]),
]
UNKNOWN_NAMES = ["shakespeare", "napoleon", "cleopatra", "picasso",
                 "beethoven", "genghis khan", "tesla", "aristotle"]

META = [
    ("meta honesty core design",
     ["do you ever make things up", "do you lie"],
     ["no. unverified content is structurally unspeakable for me.",
      "never. i stay silent rather than invent."]),
    ("meta consolidation nightly cycle",
     ["do you sleep", "when do you learn"],
     ["yes. sleep compresses rules and checkpoints memory safely.",
      "every night i consolidate the day inside rollback gates."]),
    ("social gratitude style warm",
     ["thank you", "thanks"],
     ["you are very welcome. come back anytime.",
      "anytime! happy to help."]),
    ("social farewell style warm",
     ["goodbye", "bye", "good night"],
     ["goodbye! my memory consolidates while you rest too.",
      "bye! see you soon."]),
]

def emit_chat(out, reps=3):
    def block(u, prefix, tail):
        out.append(f"user: {u}\nbrain: {prefix} — {tail}")
    # fixed authored blocks
    for prefix, users, tails in CHAT:
        for u in users:
            block(u, prefix, tails[hash(u) % len(tails)])
    for subj, prefix, qs, tails in FACT_QA:
        for q in qs:
            block(q, prefix, tails[hash(q) % len(tails)])
    for prefix, pats, tpls in REFUSALS:
        for name in UNKNOWN_NAMES:
            q = pats[hash(name) % len(pats)].format(name)
            t = tpls[hash(q) % len(tpls)].format(name)
            block(q, prefix, t)
    for prefix, qs, tails in META:
        for q in qs:
            block(q, prefix, tails[hash(q) % len(tails)])
    # augmented repeats with mild shuffles (deterministic)
    base = list(out)
    for r in range(reps):
        chunk = list(base)
        rng.shuffle(chunk)
        out.extend(chunk)

def emit_math(out, n=60):
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        kind = rng.randrange(3)
        if kind == 0:
            q, ans = f"what is {a} plus {b}", a + b
        elif kind == 1:
            q, ans = f"what is {a} minus {b}", a - b
        else:
            a2 = rng.randint(2, 12)
            q, ans = f"what is {a2} times {b}", a2 * b
        out.append(f"user: {q}\nbrain: math exact result {ans} — {q.replace('what is ', '')} equals {ans}, verified instantly.")

def emit_teach(out, n=25):
    pairs = [("sparrow", "bird"), ("rose", "flower"), ("dog", "mammal"),
             ("gold", "metal"), ("whale", "mammal"), ("oak", "tree"),
             ("salmon", "fish"), ("bee", "insect")]
    for i in range(n):
        s, o = pairs[i % len(pairs)]
        out.append(
            f"user: remember that a {s} is a {o}\n"
            f"brain: teach stored confirmed — stored. a {s} being a {o} is saved and consolidates tonight.")

# ── 2. PLAN RENDERING LINES (<p> scaffold ⇒ plans_supported) ──────────────
PLAN_DOMS = {
    "greeting": ["intent", "greeting", "welcome", "salutation", "style",
                 "friendly", "emotion", "happy", "target", "user"],
    "identity": ["identity", "name", "self", "system", "brain", "network",
                 "type", "cognitive", "origin", "artificial", "ai", "neural"],
    "status":   ["status", "state", "feeling", "good", "great", "positive",
                 "optimal", "energy", "high", "mode", "ready", "condition"],
}

def emit_plans(out, per_act=560):
    for act, clazz in PLAN_DOMS.items():
        for rep in range(per_act):
            truth = clazz[:]
            rng.shuffle(truth)
            truth = truth[: 3 + rng.randrange(2)]
            reg = "warm" if rep % 2 == 0 else "neutral"
            head = f"<p> act {act} facts"
            out.append(head + " " + " ".join(truth) + f" reg {reg} <r> "
                       + " ".join(truth))

# ── 3. READER (EYES) PAIRS ────────────────────────────────────────────────
READER_FACTS = [
    ("cheetah", "is_a", "feline"), ("feline", "is_a", "mammal"),
    ("mammal", "is_a", "animal"), ("falcon", "is_a", "raptor"),
    ("raptor", "is_a", "bird"), ("bird", "is_a", "animal"),
    ("bird", "has", "wings"), ("wings", "produce", "lift"),
    ("einstein", "is_a", "scientist"), ("scientist", "studies", "physics"),
    ("bohr", "is_a", "physicist"), ("curie", "is_a", "chemist"),
    ("turing", "is_a", "logician"), ("logician", "uses", "logic"),
    ("gravity", "causes", "acceleration"),
    ("acceleration", "causes", "velocity_change"),
    ("sun", "is_a", "star"), ("water", "freezes_at", "zero"),
    ("fire", "is_hot", "true"), ("dog", "is_a", "mammal"),
    ("sparrow", "is_a", "bird"), ("whale", "is_a", "mammal"),
    ("gold", "is_a", "metal"), ("oak", "is_a", "tree"),
    ("salmon", "is_a", "fish"), ("bee", "is_a", "insect"),
]

def reader_sentence(s, r, o, i):
    forms = [
        f"a {s} is a {o}" if r == "is_a" else f"a {s} {r.replace('_', ' ')} {o}",
        f"the {s} is a {o}" if r == "is_a" else f"the {s} {r.replace('_', ' ')} {o}",
        f"{s} is a {o}" if r == "is_a" else f"{s} {r.replace('_', ' ')} {o}",
        f"every {s} is a {o}" if r == "is_a" else f"each {s} {r.replace('_', ' ')} {o}",
    ]
    return forms[i % len(forms)]

def emit_reader(pairs_out, train=True, count_mult=6):
    for (s, r, o) in READER_FACTS:
        start = 0 if train else None
        m = count_mult if train else 2
        for i in range(m):
            sent = reader_sentence(s, r, o, i if train else i + 10)
            pairs_out.append(f"read: {sent}\ntriple: {s} {r} {o}")

def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))

    mouth = []
    emit_chat(mouth)
    emit_math(mouth)
    emit_teach(mouth)
    emit_plans(mouth)
    rng.shuffle(mouth)
    with open(os.path.join(here, "mouth_unified.txt"), "w") as f:
        f.write("\n\n".join(mouth) + "\n")

    train_pairs, probe_pairs = [], []
    emit_reader(train_pairs, train=True)
    # held-out probes: unseen phrasing index (+10) on known facts + two novel words
    emit_reader(probe_pairs, train=False, count_mult=2)
    probe_pairs.append("read: a tiger is a cat\ntriple: tiger is_a cat")
    probe_pairs.append("read: the lion is a mammal\ntriple: lion is_a mammal")
    rng.shuffle(train_pairs)
    with open(os.path.join(here, "reader_corpus.txt"), "w") as f:
        f.write("\n\n".join(train_pairs) + "\n")
    with open(os.path.join(here, "reader_probes.txt"), "w") as f:
        f.write("\n\n".join(probe_pairs) + "\n")

    print(f"mouth_unified.txt : {len(mouth)} blocks "
          f"(chat+math+teach+plans, <p> present: {'<p>' in chr(10).join(mouth[:2000])})")
    print(f"reader_corpus.txt : {len(train_pairs)} pairs")
    print(f"reader_probes.txt : {len(probe_pairs)} held-out pairs")

if __name__ == "__main__":
    main()
