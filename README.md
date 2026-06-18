# THE BRAIN (brain2)

A CPU-native cognitive system that **learns facts, reasons over them, and explains
itself** — it reads facts from text, derives conclusions it was never told, plans,
synthesizes verifiable code, consolidates memory, thinks fast-then-slow, and holds
a controlled conversation. ~150 MB, no GPU, no cloud, every answer backed by a
derivation.

It is **not** a language model and does not pretend to be. Where an LLM is strong
(fluent open-ended language, broad world knowledge) it is weak; where an LLM is
weak (learning a fact at inference and keeping it, giving a *verifiable* chain
instead of a plausible guess, running in megabytes on a laptop) it is strong.

> **Learns and reasons over knowledge on CPU, explainably — where LLMs can't, at
> a fraction of the compute.**

The active project lives in [`brain2/`](brain2/). C++ core (pybind11) + Python
engines. See [`brain2/MILESTONES.md`](brain2/MILESTONES.md) for the full capability
record and [`brain2/README.md`](brain2/README.md) for module detail.

---

## The whole loop: READ → REASON → SPEAK

```
read:   "An apple is a fruit. It is red. It grows on a tree."
            v  (fact_extractor — grammar parse, coreference)
facts:  (apple, isa, fruit) . (apple, is, red) . (apple, grows_on, tree)
            v  (knowledge + reasoning + rules + transitive closure)
ask:    "what is apple?"   -> "An apple is a fruit. It is red. It grows on a tree."
        "is it red?"       -> "Yes, apple is red."        ("it" -> apple, working memory)
        "is it blue?"      -> "Not that I know of."
        "what is banana?"  -> "I don't know anything about banana."   (honest)
```

Every word out is **derived** from a stored relation through a grammar rule —
production, not pattern-matching. It says "I don't know" when it doesn't.

## See it in 60 seconds

```bash
# build once (see Build), then from brain2/ :
python3 brain_repl.py --demo       # learn facts, derive the unstated, explain
python3 conversation_engine.py     # the full understand -> reason -> produce loop
python3 fact_extractor.py          # learn by READING text into facts
python3 tree_reason.py             # solve algebra + the bridge puzzle, show steps
python3 program_synth.py           # WRITE code from examples, verifiably
python3 dual_process.py            # reflex (System 1) + deliberation (System 2)
python3 semantic_memory.py         # memory that generalizes via meaning
```

---

## What's measured

Every claim has a number; reproduce with `python3 validate.py` (the gate).

| capability | result |
| --- | --- |
| one-shot fact retrieval | **1.0** |
| transitive inference, 5 hops, 400 distractors | **1.0** |
| noise robustness deriving 3-hop (sigma=0.5) | **0.67** (graceful — real reasoning, not lookup) |
| relation composition (parent.parent -> grandparent) | **1.0** |
| dream consolidation — catastrophic forgetting | **-73 to -84%** (auto, interleaved replay) |
| learned search guidance (8-puzzle) | **~100x** fewer states, solutions stay valid |
| program synthesis from examples | correct **by construction**, generalizes |
| dual-process recurring workload | **602 us -> 15 us** (~41x) once practiced |
| semantic memory generalization | answers "car" from a fact about "automobile" |
| footprint | ~150 MB, CPU, offline |

---

## The capability hierarchy (all hardened, gate-validated)

Built and hardened bottom-up — each a clean, tested module on the hardened layer
below it. ~130 tests, all green under `validate.py`; each rung a reversible git tag.

1. **Knowledge** — learn facts online, retrieve them (`knowledge_engine.py`)
2. **Reasoning** — composition rules across relations (`reasoning_engine.py`)
3. **General search** — branch/prune/search to a goal, optimal + deterministic (`tree_reason.py`)
4. **Knowledge + search joined** — plan over learned facts (`planning_engine.py`)
5. **Learned guidance** — search that learns to be faster from experience (`learned_guidance.py`)
6. **Verifiable synthesis** — write code from examples, correct by construction (`synthesis_engine.py`)
7. **Consolidation (dreaming)** — replay fights catastrophic forgetting (auto-replay)
8. **Dual cognition** — reflex + deliberation + compilation (`dual_process_engine.py`)

**Enhancements** (made dormant components real): **semantic memory** (the binding
memory with real embeddings — generalizes via meaning), **appraisal** (the
redesigned emotion — grades input pragmatics: question/greeting/command, tone).
**Capstone**: the **conversation loop** (`conversation_engine.py`) and **learn-by-
reading** (`fact_extractor.py`) close the read/reason/speak cycle.

## Two reasoning engines under it all

- **Binding memory** — `(subject, relation, object)` facts, online, answered by
  depth-limited transitive closure; with real embeddings it generalizes by meaning.
- **Tree search** — explore states via rule-valid operators, prune, search to a
  goal, read back the steps. The same engine solves algebra, the bridge puzzle,
  N-queens, water jugs, the 8-puzzle, symbolic rewriting, and program synthesis —
  only the operators change. It can learn its own heuristic from experience.

---

## Honest limits (the true shape of the system)

- **Language is an interface, not a strength.** Controlled conversation works
  (intent from form, grammar production). Open-domain *comprehension* of arbitrary
  phrasing and real-world meaning is the wall — that needs an LLM.
- **Operators are per-domain.** It masters the rules you give it and finds
  solutions inside them you never handed it — it does not invent new rules for a
  domain on its own.
- **It reasons over knowledge; it doesn't originate it.** Facts must be fed (by
  hand, by the grammar extractor for clean text, or by an LLM extractor for messy
  text — offline). Feed it *curated, trusted* knowledge; it is a verifiable expert,
  not a web-scale sponge.

Where an LLM fits: at the **edges**, offline where possible — reading messy text
into facts, understanding open input, fluent open output. The brain stays the
**verifiable reasoning core**; the LLM is the eyes and mouth, never the mind.

---

## Build

C++17, CMake, pybind11, and (macOS) Accelerate.

```bash
cd brain2/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8          # builds brain2.<py>.so, copied to brain2/
```

Run any demo, scorecard, or `validate.py` from `brain2/`.

## Validation (source of truth)

```bash
cd brain2
python3 validate.py          # the gate: every capability, headline result asserted (20/20)
python3 run_scorecard.py     # LM metrics (bits/char), fact retrieval, throughput
python3 reasoning_suite.py   # transitive / noise / composition reasoning
python3 tests/run_all.py     # component unit tests
```

Nothing is promoted up the hierarchy if `validate.py` regresses.
