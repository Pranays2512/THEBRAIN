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

---

## 11. Current state (updated 2026-07-02 — supersedes stale sections above)

The architecture is now **three computing pillars**, all internal, all fed by the brain's
own data; generate-and-verify with the *generate* side being fuzzy + probabilistic and the
*verify* side symbolic:

| Pillar | Role | Where |
|---|---|---|
| **Symbolic** | exact, verified, owns truth | crisp core, `means_ends`, verifiers, 7 primitives in C++ |
| **Fuzzy** | similarity, grounding, lexical meaning | SOM, `grounding`, `context_embed` |
| **Probabilistic** | distributions, uncertainty, **generation** | `prob_compute` (n-gram) → `neural_lm` / `neural_lm_torch` (MPS Transformer) |

### Self-extension (the loops that make it grow)
- **Make/break + novelty**: `refuter` (break a rule, find scope), `factorizer` (+`factor_au`
  anti-unification → grow the DSL), `concept_blend`/`ground_blend` (mint verified-novel
  concepts on the real SOM), `analogy_struct` (structure-mapping, no shared vocab).
- **Self-made verifiers**: `invariant_miner` (mine→validate→admit→demote; value + functional),
  `verifier_monitor` (audit the verifiers), `check_library` (persist across sessions),
  `synth_invariant` (pre-filter the synthesis oracle ~5×). `irregularity_detector` maps where
  verification *can't* reach (abstain).
- **Self-improving loop**: `conjecture_sandbox` (test own guesses vs trusted knowledge),
  `autonomous_loop` (curiosity→conjecture→test→bank→learn, solve-rate climbs).
- **Learned proposer** (the once-weakest part, now self-sufficient): `online_proposer` /
  `online_proposer2` / `feature_learner` — learns space-order from outcomes, transfers by
  task signature, **discovers its own predictive features**. Ported to C++ (`core/proposer.hpp`).

### C++ core ports (phase 2 — all verified == Python, locked in `harden_regress` 24/24)
`core/invariants.hpp` (inv_mine/check), `core/refuter.hpp` (refute_int1), `core/factorizer.hpp`
(eval_sexpr), `core/regularity.hpp` (law_error), `core/proposer.hpp` (disc_weights/feat_sim),
`core/reasoning_ops.hpp` (cosine_map, analogy_score). Guards: `harden_test` (35/35 no-crash) +
`harden_regress` (24/24 correctness) run before every port.

### Owned language + training (self-contained, no external model at inference)
- **Language comprehension** without an external LLM: `context_embed` (lexical, scales w/
  corpus), `structural_parser`/`nested_parser`/`deeper_grammar` (sentence shapes), and the
  learned+verified **templates** (`parse_template`, `template_memory`, conjecture→verify→admit).
  `coverage_harness` = the LM-deletion metric. `semantic_depth` = vocabulary grows from definitions.
- **Owned LM**: `neural_lm_torch` — small decoder-only Transformer on Apple-Silicon GPU (MPS).
- **One training pipeline** (`train_pipeline.py`): the qwen-coder teacher PARSES a domain into
  sentence⇒structure pairs; the **symbolic brain** learns the structure (facts + *verified*
  laws — `knowledge_distill`), the **student LM** learns the *parsing* (text→structure). Trains
  brain + student in one pass. `--real` uses the cloud teacher (bootstrap only, then disconnects).
- **Concept lifecycle**: `concept_memory` names shared structures factorizer discovers and
  promotes them by reuse (candidate→promoted).

### Open-language track (comprehension, not fluency — 2026-07-02)
Fluency is owned-LM territory (scales with corpus). *Comprehension* of open prose is won
here, crisp all the way, built in dependency order:
- **Gap 1 — richer logical form** (`event_form`): `Event(verb, agent, patient, time, polarity)`
  + typed `Relation` (CAUSE/CONTRAST/SEQUENCE). Extends, never replaces — a FACT is a
  degenerate stative Event, so the numeric core survives (`fact_as_event`/`event_as_fact`).
  Prose's negation/causality/roles/tense now have slots; more templates no longer just
  saturate a shallow shape.
- **Gap 1 membrane** (`event_verify`): the numeric gate doesn't apply to prose, so events get
  a weaker-but-crisp contract — **polarity non-contradiction** (store can't hold a claim and
  its negation) + **selectional type constraints** (a verb restricts agent/patient types).
  `type_of` is wired by `type_oracle.TypeOracle` to the **crisp `isa`-closure** of
  `core_knowledge` (dog→mammal→animal→living_thing; transitive, exact, no C++/GloVe needed).
  The disposal path (`__call__`) is **crisp-only** — fuzzy never decides admit/reject. An
  optional `similar` hook (GloVe via `build_similar_from_semantic`, off by default) instead
  feeds `grow()`: fuzzy CONJECTURES an isa edge → a verify callback (crisp/teacher) DISPOSES →
  the edge is admitted into the closure, and the token disposes exactly ever after. Fuzzy
  proposes, crisp disposes — abstention is never traded for a guess. Three-valued admit/reject/**abstain**:
  an unconstrained verb admits (numeric core vouches), a known type-violation rejects, a
  *constrained verb with an unknown role type* abstains → escalates, never guesses. Built
  BEFORE the representation on purpose — else event facts flood the truth store unverified.
- **Gap 3 — discourse** (`discourse`): coref = pointer resolution to the most recent
  type-compatible entity on a `ContextStack` (working memory, cheap); connectives
  (because/so/but/then) become typed Relations between event ids. Markers-first; implicit
  discourse stays unjudged.
- **Intake — prose → Event** (`event_parse` + `reading_loop.EventReader`): the layer that
  makes the whole membrane actually fire. `parse_event` turns a sentence into an
  `Event(verb, agent, patient, time, polarity)` — negation (`not`/`-n't`)→polarity, tense
  from aux/verb-form, SVO by nearest entity/pronoun around the verb; an unknown role token is
  SURFACED (not dropped) so the membrane can abstain on it. `EventReader` splits a sentence on
  a connective into clauses, resolves pronoun roles against the `ContextStack` (coref) BEFORE
  the membrane sees them, admits each event, and links the two with the typed Relation.
  Proven end-to-end: "cat ate fish" admits; "cat did not eat fish" then REJECTS (contradiction);
  "rock ate fish" REJECTS (type); "blorp ate fish" ABSTAINS (unknown agent); "dog chased cat
  because it was hungry" → two events + CAUSE. Markers-first/crisp (learned event templates are
  future work); coref is recency-based (salience is a known limitation).
- **Positional verb detection** (`event_parse` step 2 + `verb_trusted`): a sentence with no
  trusted verb still parses — the verb is taken as the first content token after the subject,
  recovering SVO structure on *unknown* verbs (government/**raised**/interest, engineers/
  **deployed**/patch). But an untrusted verb makes the event ABSTAIN, never admit: structure
  reaches the membrane, the truth store stays clean. This took **wild-prose parse coverage
  0% → 100%** (`coverage_harness.event_coverage_split`), every wild event correctly held
  (abstain, not asserted). Moving wild abstain→admit is the next depth: verb acquisition
  (grow selectional constraints per verb — the `type_oracle.grow` pattern applied to verbs).
- **Gap 2 — autonomous reading** (`reading_loop`): parse→Event→verify→admit, and only ADMIT
  parses (+ trusted teacher labels) re-enter template induction (**anti-collapse gate**).
  Grammar misses escalate **per-fragment** to the teacher, buffer per-relation, induce once
  ≥2 exist (real anti-unify + held-out check). Escalation rate is tracked — the honest
  "is the teacher still needed" decay signal.
- **Gap 4 — honest metric** (`coverage_harness.coverage_split`): taught-domain coverage
  FLATTERS (grammar was fitted there). `wild` = held-out text outside taught domains — the
  real open-language number; the reported `gap` stops the flattering figure being quoted alone.
- Reject-tests: `test_open_lang.py` 29/29 (membrane rejects contradiction/type-violation,
  abstains on the unknown, admits the good; coref/connective direction; escalation decay).

### The verifying agent
`agent.py` — plan → act → **verify each step** → self-correct → repeat. Success stays flat over
horizon (5–40 steps: 1.00) where an unverified agent decays (0.95^n → 0.13). Reliability beats
parameters where errors compound.

### Honest ceilings (current)
- Problem-solving (math/code/physics/law-discovery) is **reliable now** via the symbolic core,
  independent of LM scale; it **abstains** outside the verifiable envelope.
- Open-language *fluency* scales with the owned LM's corpus + size (small owned LM + symbolic
  offload = reliable, not frontier-fluent). External LLM = one-time bootstrap teacher.
- Dimensional/other hard filters use the three-valued True/False/**None** contract — never
  prune what they don't understand.
- Remaining C++ perception-substrate gaps are logged in `architecture_flaws.md` (deferred, not
  bugs in the path the current system runs).
