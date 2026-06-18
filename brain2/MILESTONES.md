# brain2 — capability milestones

The development checkpoints the brain has reached, lowest to highest. Each is
validated by a script in this repo (run `python3 validate.py` to re-check them
all — the promotion gate). New capabilities are prototyped, validated here, then
promoted up the hierarchy. Nothing moves up if `validate.py` regresses.

| # | milestone | what it is | validated by | result |
|---|-----------|------------|--------------|--------|
| 1 | **Knowledge** ✅ HARDENED | learn facts online, retrieve them | `knowledge_engine.py`, `tests/test_knowledge_engine.py` | one-shot retrieval 1.0; 17 hardening tests green |
| 2 | **Reasoning** ✅ HARDENED | composition rules across relations, explained | `reasoning_engine.py`, `tests/test_reasoning_engine.py` | parent∘parent→grandparent + nested rules; 12 tests green |
| 3 | **General search** ✅ HARDENED | branch / prune / search to a goal over rules | `tree_reason.py`, `tests/test_search_engine.py` | optimal + deterministic; algebra, bridge=17, N-queens, jugs, rewrite; 16 tests |
| 4 | **Knowledge + search joined** ✅ HARDENED | plan over the facts it learned | `planning_engine.py`, `tests/test_planning_engine.py` | multi-precondition planning, optimal, online replan; 12 tests |
| 5 | **Learned search guidance** ✅ HARDENED | learns to search more efficiently from experience | `learned_guidance.py`, `tests/test_learned_guidance.py` | LearnedHeuristic ~100× fewer states, solutions stay valid; 9 tests |
| 6 | **Verifiable program synthesis** ✅ HARDENED | write code from examples, correct by construction | `synthesis_engine.py`, `tests/test_synthesis_engine.py` | verified on spec, generalizes, shortest program, honest failure; 13 tests |
| 7 | **Consolidation (dreaming)** ✅ HARDENED | replay fights catastrophic forgetting, automatic | `tests/test_consolidation.py` | auto-replay cuts A forgetting 1.99→0.54 (~73%), new task kept; 4 tests |
| 8 | **Dual cognition** ← **CURRENT CHECKPOINT** | reflex (System 1) + deliberation (System 2), with compilation | `dual_process.py` | recurring workload 602 µs → 15 µs (~41×); deliberation compiles into reflex |

## Honest boundary (true at every milestone above)

- Operators are defined per domain — it masters rules you give it, doesn't invent them.
- It reasons over knowledge; it doesn't originate knowledge from raw perception.
- Language is a capacity-bound interface, not a strength.

## Hardening status (bottom-up)

Capabilities are hardened from the bottom of the hierarchy up — each becomes a
clean, validated, persistent module before the next is promoted.

- **Infra** ✅ — `validate.py` gate + this checkpoint record.
- **#1 Knowledge** ✅ — `knowledge_engine.py`: clean API (`learn`/`ask`/`derive`/
  `explain`/`knows`), input validation, idempotency, cycle handling, distractor
  robustness, JSON persistence, 17 tests. (Hardening caught and fixed a real
  bug: a strong relation match could compensate for an absent subject, giving
  dead-end false positives — fixed with a stricter confidence threshold.)
- **#2 Reasoning** ✅ — `reasoning_engine.py`: composition rules across
  different relations (`X parent Y & Y parent Z => X grandparent Z`), backward
  chaining, nested rules, transitive relations, cycle-safe, explained, persisted.
  12 tests. Builds on the hardened KnowledgeEngine — the rule layer above
  fact retrieval.
- **#3 General search** ✅ — `tree_reason.py` hardened: documented guarantees
  (optimal with non-negative costs + admissible heuristic, deterministic,
  cycle-terminating), input validation (rejects negative costs, bad max_nodes,
  malformed problems), clean no-solution / node-cap handling, ergonomic
  `search()` -> SearchResult. 16 tests across all domains.
- **#4 Knowledge + search joined** ✅ — `planning_engine.py`: the first layer
  built on TWO hardened layers (KnowledgeEngine stores actions as facts; the
  hardened search plans over them). Multi-precondition planning, optimal/minimal
  plans, online replan when actions are added, distractor-robust, validated,
  persisted, explained. 12 tests. (This is why bottom-up order matters — it
  reuses already-proven layers.)
- **#5 Learned search guidance** ✅ — `learned_guidance.py`: a reusable
  `LearnedHeuristic` (domain-agnostic, fit by least squares over state features
  from solved instances) that guides the hardened search engine. Tests pin the
  two properties that matter — guided solutions stay VALID, and search expands
  ~100× fewer states than blind — plus determinism, persistence, validation.
  9 tests. (The synthesis policies — guided/policy/tree — remain validated by
  `validate.py --full`.)
- **#6 Verifiable program synthesis** ✅ — `synthesis_engine.py`: clean
  `SynthesisEngine` on the hardened search. Examples in -> shortest program that
  reproduces them all (correct BY CONSTRUCTION), generalizes to unseen inputs,
  fails honestly outside the DSL. Tests pin verification-on-spec, generalization,
  shortest-program, determinism, and validation. 13 tests.
- **#7 Consolidation (dreaming)** ✅ — lives in the C++ training loop
  (auto-replay rehearses recent sequences while learning new ones). Hardened
  by a focused test pinning the production guarantee: auto-replay cuts
  catastrophic forgetting of an earlier task by ~73% (learn A -> learn B ->
  retest A) WITHOUT sacrificing the new task, and the replay knobs are honored.
  4 tests.
- **#8 Dual cognition** — last rung; promote next, gate-guarded.

## Workflow

1. **Prototype** a new capability (Python, fast iteration).
2. **Validate** it — add a check to `validate.py`, confirm the gate stays green.
3. **Promote** — once proven and stable, harden toward a real target / move into
   the C++ core for speed. One capability at a time; the scorecard + gate guard
   against regressions.

The C++ brain state persists via `save_components` / `load_components`
(round-trips cleanly). This file is the *capability* checkpoint — what the brain
can do and the evidence for it.
