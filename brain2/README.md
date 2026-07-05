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
python3 dual_process.py            # reflex (System 1) + deliberation (System 2), 40x on recurring work
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
- **Origination is bounded, not absent.** The brain *does* originate: it
  conjectures laws and self-tests them against principles it already trusts
  (`conjecture_sandbox`), runs a standing curiosity→conjecture→test→bank→learn
  cycle that rediscovers laws with no answer key and transfers a learned shape to
  new gaps (`autonomous_loop`), and unifies two distant domains under one shared
  law via anti-unification (`curiosity_cross`, e.g. ½·m·v² and ½·k·x² collapse to
  ½·coeff·quantity²). The **boundary**: conjectures are drawn from a form grammar
  (compositions of known operators) and must have a trusted anchor to test against.
  So it originates *novel-but-related* knowledge, not *zero-precedent* structure —
  the same boundary a scientist has. It is not a passive store.
- **Wiring lags capability.** Several of the origination/creativity faculties above
  are BUILT and pass their own demos but are **not yet wired into the `whole_brain`
  front** — so a live `ask()` today does routing→facts→laws→ISA→synth→arithmetic,
  not curiosity/blend/cross-domain. See §"Module wiring status" for the wired-vs-
  orphan map. Making these live is integration work, not training.
- **Operators are still per domain.** It reasons brilliantly inside rules you give
  it and finds solutions you never handed it (the bridge-puzzle optimum), and the
  autonomous loop discovers laws within its form grammar — but it does not invent a
  genuinely new operator outside that grammar on its own.

These aren't bugs to hide — they're the true shape of the system.

---

## Module wiring status (as of 2026-07-05)

Import-reachability from the live entry points (`whole_brain`, `exam`, `train_pipeline`,
`reading_loop`, `autonomous_loop`, `nl_front`): **126 modules, 70 wired (56%).** A passing
`__main__` demo is not a wire — the system's real behavior is only what the front reaches.

**Wired into the live `ask()` path:** `reasoning_engine`, `means_ends`, `synth_engine`,
`brain_store`, `appraisal_engine`, `reading_loop`, `type_oracle`, `verb_learn`,
`event_*`, `mouth`, `synth_invariant`, `verifier_monitor`, `autonomous_loop`,
`context_embed`, `check_library`, `+ C++ core`, and the **creativity faculties** (below).

**Creativity / origination — NOW WIRED** (`whole_brain` methods, membrane-gated):
`curiosity_cross` (`cross_domain()`), `concept_blend` (`blend()`), `analogy_engine`
(`analogize()`), `inductive_engine` (`induce()`), `learn_by_reading` (`read_to_law()`),
composed in `create()`. The front reads episodes → originates verified rules → uses them.

**Still orphaned (built + run, NOT wired):**

| Group | Modules | Note |
|---|---|---|
| B. Alternate fronts | `brain_chat`, `chat`, `brain_repl`, `brain_session`, `math_chat`, `server`, `agent`, `brain_planner`, `planning_engine` | never chosen as THE front |
| C. Grounding | `ground_blend`, `ground_numeric`, `ground_reason`, `ground_to_binding`, `crispify_bridge` | perception→symbol, disconnected |
| D. Synthesis variants | `program_synth`, `synthesis_engine`, `loop_synth`, `loop_synth2`, `dp_greedy_synth`, `stress_synth`, `compositional` | likely superseded by `synth_engine`; audit + prune |
| E. Search accelerators | `learned_guidance`, `composable_proposer`, `online_proposer2` | speed search; need synth backends to expose a prior hook (distinct refactor, not an ask()-path capability) |
| F. Language / memory | `semantic_memory`, `conceptnet_taxonomy`, `knowledge_pack`, `dual_process`, `dual_process_engine` | disconnected |

Regenerate: import-reachability BFS from the entry points over local imports
(see the graphify graph at `../graphify-out/`, `graphify update .`).

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
