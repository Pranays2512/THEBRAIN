# brain2 — capability milestones

The development checkpoints the brain has reached, lowest to highest. Each is
validated by a script in this repo (run `python3 validate.py` to re-check them
all — the promotion gate). New capabilities are prototyped, validated here, then
promoted up the hierarchy. Nothing moves up if `validate.py` regresses.

| # | milestone | what it is | validated by | result |
|---|-----------|------------|--------------|--------|
| 1 | **Knowledge** ✅ HARDENED | learn facts online, retrieve them | `knowledge_engine.py`, `tests/test_knowledge_engine.py` | one-shot retrieval 1.0; 17 hardening tests green |
| 2 | **Reasoning** ✅ HARDENED | composition rules across relations, explained | `reasoning_engine.py`, `tests/test_reasoning_engine.py` | parent∘parent→grandparent + nested rules; 12 tests green |
| 3 | **General search** | branch / prune / search to a goal over rules | `tree_reason.py`, `tree_domains.py` | algebra, bridge puzzle, N-queens, water jugs, rewriting — all solved |
| 4 | **Knowledge + search joined** | plan over the facts it learned | `brain_planner.py` | multi-precondition planning (forge needs iron AND wood) |
| 5 | **Learned search guidance** | learns to search more efficiently from experience | `tree_learn.py`, `program_synth_*` | heuristic 127× fewer states; synthesis policy 3.8×; tree policy 5.4× |
| 6 | **Verifiable program synthesis** | write code from examples, correct by construction | `program_synth.py` | synthesizes + generalizes; honest failure outside the DSL |
| 7 | **Consolidation (dreaming)** | replay fights catastrophic forgetting, automatic | `component_validation.py` | −84% forgetting (interleaved faithful replay) |
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
- **#3 and up** — still prototype-grade; promote next, gate-guarded.

## Workflow

1. **Prototype** a new capability (Python, fast iteration).
2. **Validate** it — add a check to `validate.py`, confirm the gate stays green.
3. **Promote** — once proven and stable, harden toward a real target / move into
   the C++ core for speed. One capability at a time; the scorecard + gate guard
   against regressions.

The C++ brain state persists via `save_components` / `load_components`
(round-trips cleanly). This file is the *capability* checkpoint — what the brain
can do and the evidence for it.
