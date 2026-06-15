# brain2

A CPU-native system that **learns facts online and reasons over them** — two ways:
deductive closure over relations, and goal-directed search — and **shows its work**.
It runs in ~150 MB on a laptop CPU, with no GPU and no cloud.

It is **not** a language model and does not try to be. Where an LLM is strong
(fluent open-ended language) it is weak; where an LLM is weak (learning a new
fact at inference and keeping it, giving a verifiable derivation instead of a
plausible guess) it is strong. The honest one-line pitch:

> **Learns knowledge continuously and reasons over it on CPU, explainably —
> where LLMs can't, at a fraction of the compute.**

---

## See it in 30 seconds

```bash
# (build once — see "Build" below, then from this directory:)
python3 brain_repl.py --demo       # learn facts, derive the unstated, explain
python3 tree_reason.py             # solve algebra + a planning puzzle, show steps
python3 brain_planner.py           # knowledge + reasoning joined: plan over learned facts
python3 tree_learn.py              # search that LEARNS its heuristic (127x fewer states)
python3 tree_domains.py            # same engine: N-queens, water jugs, symbolic rewriting
python3 program_synth.py           # WRITE code from examples, by search, verifiably
python3 program_synth_guided.py    # synthesis that LEARNS which operators to try
python3 program_synth_policy.py    # goal-conditioned policy guides each next op (3.8x)
python3 program_synth_tree.py      # the policy as a decision tree (5.4x, interpretable)
```

**Write a program from examples, by search** (`program_synth.py`) — it searches a
DSL for a program that reproduces the examples, then generalizes to new inputs.
Every program returned is correct on the spec *by construction*, not guessed:

```
"John Smith" -> "JOHN"     SYNTHESIZED:  upper -> first_word
"bob dylan"  -> "BOB"      "alice cooper" -> "ALICE"   (generalizes; never shown)
```

**Learn online, derive what you never stated, explain the chain** (`brain_repl.py`):

```
> alice > bob          (teach links one at a time)
> bob > carol
> carol > dave
> alice > ?
  alice > dave  — DERIVED, never told. Here's how:
  alice  >  bob  >  carol  >  dave
```

**Reason to a goal over its own learned facts** (`brain_planner.py`):

```
Teach:  smelt requires ore / produces iron
        chop  requires axe / produces wood
        forge requires iron + wood / produces sword
Goal:   sword from ore + axe
  1. chop:  axe  -> wood
  2. smelt: ore  -> iron
  3. forge: iron + wood -> sword     (needs iron AND wood: real planning)
```

---

## What's measured

Every claim has a number behind it. Reproduce with the scorecards below.

| capability | result |
| --- | --- |
| one-shot fact retrieval (binding) | **1.0** |
| transitive inference, up to 5 hops, with 400 distractors | **1.0** |
| noise robustness deriving 3-hop (σ=0 / 0.2 / 0.5 / 1.0) | 1.0 / 0.88 / **0.67** / 0.38 |
| relation composition (parent∘parent → ancestor) | **1.0** |
| dream consolidation — catastrophic forgetting reduced | **−84%** (auto, interleaved replay) |
| algebra + bridge-puzzle planning | solved, optimal, steps shown |
| language model (held-out bits/char) | ~1.72 — **capacity-bound, not the point** |
| footprint / throughput | ~150 MB, ~900 words/sec, CPU |

The noise-robustness number is the important one: brittle table-lookup collapses
at σ=0.2; this degrades gracefully, so it is **real similarity-based reasoning,
not memorized retrieval**.

---

## The two reasoning engines

1. **Binding memory** — stores `(subject, relation, object)` facts, taught online,
   and answers queries by **depth-limited transitive closure**. `A>B, B>C ⟹ A>C`,
   derived and explained. Fast, special-purpose relational deduction.

2. **Tree search** (`tree_reason.py`) — the general engine: explore states by
   applying rule-valid **operators**, prune dead branches, search to a **goal**,
   read back the path as worked steps. Transitive closure is just its simplest
   case. The same `solve()` engine handles linear algebra, the bridge-and-torch
   puzzle, N-queens, water jugs, the 8-puzzle, and symbolic rewriting — only the
   operators change. It can **learn its own search heuristic** from solved
   experience (`tree_learn.py`: ~127× fewer states than blind search), and
   `brain_planner.py` joins the two engines — it plans over facts the binding
   memory learned.

Supporting parts (C++ core): a self-organizing map for concept grounding, an
LSTM predictor for sequence modeling, episodic + working memory, and a **dream
consolidation** cycle (interleaved experience replay) that cuts forgetting 84%.

---

## Honest limits

- **Language is capacity-bound.** Held-out bits/char is ~1.72 regardless of data
  scale; more epochs make it worse. A 2-layer LSTM at 64-d frozen embeddings will
  never be LLM-class, and that's fine — language is the interface, not the value.
- **Operators are defined per domain.** It reasons brilliantly *inside* rules you
  give it — and finds solutions you never handed it (it discovered the bridge
  puzzle's counterintuitive optimum) — but it does **not invent new rules or
  operators** for a domain on its own. Each new problem class is a plug-in.
- **It does not originate knowledge.** It must be told relations (or fed them by
  an extractor); it reasons over knowledge, it doesn't generate it from raw
  perception.

These aren't bugs to hide — they're the true shape of the system.

---

## Build

Requires a C++17 compiler, CMake, pybind11, and (on macOS) Accelerate.

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8          # produces brain2.<py>.so, copied to the brain2/ root
```

Then run any demo or scorecard from the `brain2/` directory.

## Scorecards (the source of truth — nothing merges if these regress)

```bash
python3 run_scorecard.py            # language-model metrics (bits/char), fact retrieval, throughput
python3 reasoning_suite.py          # transitive / noise / composition reasoning
python3 component_validation.py     # ablation: dream (84%) and emotion, each on its own axis
python3 tests/run_all.py            # component unit tests
```

`run_scorecard.py` writes `scorecard.json` and compares against the committed
`scorecard_baseline.json`.
