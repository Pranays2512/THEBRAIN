# Brain v2 — Architecture & Data Flow

Brain v2 is a **neurosymbolic** system: a fuzzy neural substrate (perception,
grounding, associative memory, language) joined to a **crisp symbolic reasoner**
(facts, composable policies, program synthesis) by a strict **verifier** discipline.
The thesis: *the neural/LLM side reads and guides; the symbolic side composes and
proves.* The brain owns truth + reasoning + verified code; an LLM is a thin,
shrinking language shell. The win is reliability, auditability, and efficiency on a
bounded-but-growing domain — not general LLM parity.

> The earlier predictive-coding brain (SOM → Predictor → Language) is still here —
> it is the **perception substrate** (§6), now one part of a larger system.

---

## 1. The two halves and the membrane

| | Fuzzy / neural half (C++) | Crisp / symbolic half (C++ + Python) |
|---|---|---|
| Stores | `BindingMemory` (vector triples), SOM | `PolicyEngine`/`PolicyMemory`, fact graph |
| Recall | similarity + **confidence** | exact lookup + composition |
| Strength | perception, generalization, language | exact computation, proof, code |
| Failure | approximate (a near-but-wrong vector) | none (verified or honest "unknown") |

**The membrane** is the load-bearing rule: generalization may *propose*; only the
crisp store *answers*. They communicate through a **confidence gate**
(`crispify_bridge.py`: a BindingMemory vector recall ≥0.9 → a crisp fact), never by
merging. Fuzzy recall corrupting the truth store is the failure this prevents.

---

## 2. Faculties

- **Perceive & ground** — raw vectors → SOM concepts → grounded symbols; grounded
  category *or* numeric value flows into reasoning (`grounding.py`, `ground_reason.py`,
  `ground_numeric.py`, `ground_to_binding.py`). Recognition is itself verified (held-out
  accuracy).
- **Reason** — means-ends executive over facts + composable policies, with a
  **proposer** (premise selection) and conjecture→verify→admit learning
  (`means_ends.py`, `policy_proposer.py`; ported to C++ `core/policy_engine.hpp`,
  embedded in the Brain as `brain.policy_solve/learn`).
- **Discover laws** — guided symbolic regression from data, held-out + dimensional
  verified (`policy_induction.py`, `dimensional_verify.py`); from text via an LLM
  reader (`learn_by_reading.py`).
- **Synthesize algorithms** — a ladder of program spaces, proposer-guided,
  stress-verified (§4).
- **Language front** — escalation ladder lexical → distilled student → local LLM →
  honest "I don't know" (`nl_front.py`, `student_trainer.py`).
- **Remember** — verified knowledge persists and accumulates across sessions
  (`brain_store.py`).

---

## 3. The one front — `whole_brain.py`

`ask(text)` routes a request and returns `(route, answer, verified)`:

```
            ┌─ COMPUTE → means-ends executive (facts + policies)      → verified number
ask(text) ──┼─ FACTUAL → ReasoningEngine (transitive isa + inherit)   → verified yes/no
            ├─ CODE    → synth_engine (verified) OR recall from store → verified code
            └─ UNKNOWN → honest "I don't know"
```

Routing order matters: compute is checked before the loose factual handlers; is-a
fires only among *known concepts* (out-of-vocab → honest unknown, not a wrong "No").
Persistence threads through: ask for the same function twice → the second is recalled,
not re-derived.

---

## 4. The universal engine pattern

Every search in the system — physics formulas, programs, DP recurrences,
compositions — runs the **same three-part engine**:

```
composable pieces  →  learned PROPOSER (premise selection from features)  →  VERIFIER gate
```

- **Synthesis ladder** (all LLM-free, each verified): formulas (`brain_codegen`) →
  folds (`loop_synth`) → two-state+conditional (`loop_synth2`) → while+early-return
  (`loop_synth3`) → list+nested (`loop_synth4`) → DP/greedy (`dp_greedy_synth`) →
  **composable** (`composable_synth`, one space whose primitives recombine into novel
  algorithms).
- **Proposers** order the search by I/O features → ~4.5–5.4× fewer evaluations
  (`composable_proposer`, `dp_proposer`, `program_synth_tree`).
- **One entry**: `synth_engine.py` routes a task to the applicable spaces; benchmark =
  match real Python functions, **gated by stress-vs-oracle (1000 random inputs)**.

---

## 5. Verifiers (the reach = the generality)

A problem is solvable iff it (a) parses to a goal, (b) decomposes into known steps,
(c) is **verifiable**. The verifier suite:

| Domain | Gate |
|---|---|
| Symbolic math | substitute-back / differentiate-back |
| Induced formulas | held-out fit **+** dimensional units |
| Perception | recognition accuracy (held-out) |
| Parsing | confidence floor + abstain |
| **Code / algorithms** | **stress-vs-oracle, in the synthesis loop** (`stress_synth.py`) |

Stress-in-the-loop self-corrects overfits with no human help (the synthesizer skips a
program that fits the examples but fails random cases). Growing the verifiers grows
the verifiable envelope — the honest path toward generality.

---

## 6. Perception substrate (the original neural brain)

Still real, still C++ (`core/brain.hpp` + components). The **Perceive → Think → Speak**
loop: raw vector → **SOM** (BMU + activation map, `som.hpp`) → **Predictor** (LSTM,
prediction error = surprise) → **Emotion** (valence/arousal modulate learning) →
**Attention** (salience gate) → **Working/Episodic Memory**. Grounded **Language** maps
SOM activations to words. `find_bmu` is brute-force EXACT on purpose ("correctness
first") — the fast LSH/greedy path is approximate and corrupts SOM training.

Sanity values: `n_dims` 64–256, `som_rows/cols` 16–64, `prediction_error` 1.0→<0.1.

---

## 7. Persistence

`brain_store.py` accumulates verified policies, facts, and synthesized functions as
JSON (gitignored), reloaded each run; the C++ Brain persists policies+facts via
`save_components` (`policies.txt`/`facts.txt`). Nothing enters a store unverified, so
accumulation cannot drift into confident-wrong knowledge.

---

## 8. Honest ceilings

- **Not general** — verification bounds it; open/creative/unverifiable questions →
  borrow from the LLM or abstain. Reachable: *general within verifiability*.
- **Synthesis** covers the algorithm zoo (incl. composed novelty), not arbitrary
  programs (no recursion/nested-search beyond templates); novel research-level
  recurrences need insight that example-fit overfits.
- **Greedy** correctness needs a brute oracle, not examples (stress catches traps).
- **Scale**: graph reasoning scales (100k facts, ~0.01 ms/query after the O(N²)→O(N)
  load fix); fuzzy binding-memory recall is capped; brute SOM `find_bmu` walls ~1M
  neurons.
- **Language periphery** (fluency, open phrasing) stays the LLM's; the distilled
  student (≈82% held-out) shrinks the LLM's runtime role but not to zero.

---

## 9. Environment & build

- C++ core builds via `build/` (~30s); `import brain2` works under **venv2's** python
  (the `.so`'s ABI). venv2 has numpy + sympy + brain2 → full stack in one process.
- Run brain2-dependent scripts with `venv2/bin/python3`.
- Roadmap + status: `architecture_roadmap.md`. Real `--real` LLM paths need the Ollama
  server up (teacher = `qwen3-coder:480b-cloud`).

---

## 10. File map (selected)

| File | Role |
|---|---|
| `whole_brain.py` | one front: compute / factual / code / honest unknown |
| `means_ends.py`, `core/policy_engine.hpp` | crisp reasoner (Python + native C++) |
| `policy_induction.py`, `learn_by_reading.py` | discover laws (data / text) |
| `synth_engine.py` + `*_synth*.py` | unified program synthesis + spaces |
| `composable_proposer.py`, `dp_proposer.py` | learned search guidance |
| `nl_front.py`, `student_trainer.py` | language ladder + distilled parser |
| `grounding*.py`, `ground_*.py` | perception → grounded symbols/numbers |
| `crispify_bridge.py` | the fuzzy→crisp confidence gate |
| `brain_store.py` | cross-session verified accumulation |
| `conceptnet_taxonomy.py`, `scale_test.py` | real-knowledge + scale |
| `core/brain.hpp` + components | perception substrate (SOM/predictor/…) |
