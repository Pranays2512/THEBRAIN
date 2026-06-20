# THE BRAIN (brain2)

A CPU-native **verifiable reasoning core** — it learns facts, reasons over them,
does symbolic math, physics and code, discovers rules from data, and **explains or
proves every answer**. Designed so an LLM is only the *eyes and mouth*: it
translates language in and out, while the brain stays the mind. ~150 MB, no GPU,
no cloud.

It is **not** a language model. Where an LLM is strong (fluent open language, broad
knowledge) the brain leans on the LLM as a peripheral; where an LLM is weak
(learning a fact at inference, giving a *verifiable* chain instead of a plausible
guess, saying "I don't know" instead of bluffing, running in megabytes) the brain
is the answer.

> **Right whenever it commits, honest otherwise — at a fraction of the compute.**

The active project lives in [`brain2/`](brain2/). C++ core (pybind11) + Python
engines. Full capability record in [`brain2/MILESTONES.md`](brain2/MILESTONES.md).

---

## The product architecture: LLM = eyes & mouth, brain = brain

```
text --Eyes(LLM)--> Query --BRAIN(controller)--> Answer --Mouth(LLM)--> text
                              reason · know · math · verify · discover
```

The brain operates **only on structured Query/Answer** — it never sees raw text.
The LLM is one swappable implementation of `Eyes`/`Mouth` ([`neuro_bridge.py`](brain2/neuro_bridge.py));
because it only translates, **it cannot invent facts**. Every answer's content
comes from the verified symbolic core. Proven swappable: the same brain `Answer`
renders through different mouths unchanged.

**The metric a bare LLM can't report** ([`eval_harness.py`](brain2/eval_harness.py)):

| metric | value |
| --- | --- |
| coverage (answered vs honestly declined) | 83% |
| accuracy on answered | 100% |
| **verified-correct rate** | **100%** — right whenever it says *verified* |

It is graded on *calibration*, not bluffing: honest declines ("no elementary
integral", "I don't know X") count as correct. A frontier LLM at 88% MMLU still
can't tell you *which* answers to trust; this can.

## Run the whole brain

```bash
# build once (see Build), then from brain2/ :
python3 brain_session.py           # boot knowledge, ask anything (knowledge + math), ingest live
python3 math_chat.py               # "differentiate sin(x^2)" / "solve 2*x+3=7 for x"
python3 discovery.py               # learn rules from data + analogy, verify, reason
python3 eval_harness.py            # the trust metrics
python3 validate.py                # the gate: 42 checks, every headline result asserted
```

```
$ python3 brain_session.py
> what is a dog?          A dog is a canine, a domestic animal and a pet. It can bark, bite...
> differentiate sin(x^2)  The derivative is cos(x^2)*(2*x).
> what is a whale?        I don't know anything about whale.
> :ingest whales.txt      +1 fact
> what is a whale?        A whale is a mammal.
```

---

## What it reasons over

**Language** — read → reason → speak, fully derived (no pattern-matching):
```
read:  "An apple is a fruit. It is red. It grows on a tree."  -> (apple,isa,fruit) ...
ask:   "what is apple?"   -> "An apple is a fruit. It is red. It grows on a tree."
       "is a dog an animal?" -> "Yes — dog -> pet -> animal."   (multi-hop, derived)
       "how does fruit grow?" -> "Sunlight leads to photosynthesis, which leads to ..."
```

**Math** — symbolic, correct by construction, every result verified:
```
differentiate sin(x^2)  -> cos(x^2)*(2*x)     (chain + trig + power composed)
integrate cos(x)        -> sin(x) + C          (verified by differentiating back)
integrate sin(x^2)      -> "no elementary form" (honest, not faked)
solve 2*x + 3 = 7       -> x = 2               (verified by back-substitution)
```

**Physics** — apply a law, solve for ANY variable: `F=m*a, given F,m -> a = F/m`.

**Coding** — one spec → idiomatic Python / C++ / Java class boilerplate (the Python
output compiles and runs in the tests).

**Discovery** — learns rules instead of only being told them:
- **induction** — mine "B follows A" from data, **verify on a hold-out**, promote
  only what replicates (rejects coincidences like *cat → rainbow*).
- **analogy** — structure-map one domain onto another (*pump:battery, flow:current*
  → predicts *battery raises voltage*).
- **curiosity** — when idle, hunt high-prediction-error gaps and fill them;
  stays honestly curious where no stable rule exists.

All proposals pass the **same verify-before-promote gate** — propose freely, trust
nothing unverified.

---

## What's measured

Every claim has a number; reproduce with `python3 validate.py` (the gate, **42/42**).

| capability | result |
| --- | --- |
| one-shot fact retrieval | **1.0** |
| transitive inference, 5 hops, 400 distractors | **1.0** |
| multi-parent category closure (dog → pet → animal) | derived, chain shown |
| symbolic differentiation | correct by construction, numerically verified |
| integration | verified by differentiating back; honest `None` outside ruleset |
| algebra / physics solve | verified by back-substitution |
| inductive rule learning | spurious rules rejected on hold-out |
| dream consolidation — catastrophic forgetting | **−73 to −84%** (auto replay) |
| verified-correct rate (eval harness) | **100%** |
| footprint | ~150 MB, CPU, offline |

---

## The capability stack (all gate-validated)

**Foundations** (hardened bottom-up): Knowledge → Reasoning → General search →
Planning → Learned guidance → Verifiable synthesis → Consolidation → Dual cognition.

**Reasoning core** — two engines: a **binding memory** of `(subject, relation,
object)` facts (online, transitive closure, generalizes by meaning with real
embeddings), and a **tree search** that solves algebra, puzzles, rewriting and
program synthesis by changing only the operators, and can learn its own heuristic.

**Math/physics/coding** — calculus, integral, algebra, physics, code-gen engines
over a shared exact expression representation + a recursive-descent notation parser.

**Discovery** — induction, analogy, curiosity, unified in one propose→verify→reason
cycle.

**Product spine** — `neuro_bridge` (Eyes/Brain/Mouth contract), `knowledge_base`
(ingest + dedupe + a measurable coverage dial), `eval_harness` (trust metrics),
`brain_session` (a bootable, ingestable, queryable brain).

---

## Honest limits (the true shape)

- **Language is an interface, not a strength.** Controlled NL + exact math notation
  work; open-domain comprehension of arbitrary phrasing is the wall — that's the
  LLM-as-eyes job.
- **Coverage = what it knows.** It answers from fed knowledge or says "I don't
  know" — no graceful bluffing. Broad benchmarks (MMLU) need the
  knowledge-ingestion grind, tracked by `knowledge_base` coverage.
- **It reasons; it doesn't invent without grounds.** Discovery proposes (mining,
  analogy) but promotes nothing that fails verification. No causation without
  intervention, no open-ended invention, no hard integrals / theorem rediscovery
  (that's the exponential frontier).

The LLM fits at the **edges** — reading messy text into structure, fluent output —
constrained to the brain's verified content. **The brain is the mind; the LLM is
the eyes and mouth.**

---

## Build

C++17, CMake, pybind11, and (macOS) Accelerate.

```bash
cd brain2/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8          # builds brain2.<py>.so, copied to brain2/
```

## Validation (source of truth)

```bash
cd brain2
python3 validate.py          # the gate: 42 checks, every headline result asserted
python3 tests/run_all.py     # component unit tests
```

Nothing is promoted if `validate.py` regresses.
