# brain2 — capability milestones

The development checkpoints the brain has reached, lowest to highest. Each is
validated by a script in this repo (run `python3 validate.py` to re-check them
all — the promotion gate). New capabilities are prototyped, validated here, then
promoted up the hierarchy. Nothing moves up if `validate.py` regresses.

| # | milestone | what it is | validated by | result |
|---|-----------|------------|--------------|--------|
| 1 | **Knowledge** ✅ HARDENED | learn facts online, retrieve them | `knowledge_engine.py`, `tests/test_knowledge_engine.py` | one-shot retrieval 1.0; 17 hardening tests green |
| 2 | **Reasoning** ✅ HARDENED | composition rules + multi-parent transitive closure, explained | `reasoning_engine.py`, `tests/test_reasoning_engine.py` | parent∘parent→grandparent + nested rules + DAG closure (dog→pet→animal over MANY parents); tests green |
| 3 | **General search** ✅ HARDENED | branch / prune / search to a goal over rules | `tree_reason.py`, `tests/test_search_engine.py` | optimal + deterministic; algebra, bridge=17, N-queens, jugs, rewrite; 16 tests |
| 4 | **Knowledge + search joined** ✅ HARDENED | plan over the facts it learned | `planning_engine.py`, `tests/test_planning_engine.py` | multi-precondition planning, optimal, online replan; 12 tests |
| 5 | **Learned search guidance** ✅ HARDENED | learns to search more efficiently from experience | `learned_guidance.py`, `tests/test_learned_guidance.py` | LearnedHeuristic ~100× fewer states, solutions stay valid; 9 tests |
| 6 | **Verifiable program synthesis** ✅ HARDENED | write code from examples, correct by construction | `synthesis_engine.py`, `tests/test_synthesis_engine.py` | verified on spec, generalizes, shortest program, honest failure; 13 tests |
| 7 | **Consolidation (dreaming)** ✅ HARDENED | replay fights catastrophic forgetting, automatic | `tests/test_consolidation.py` | auto-replay cuts A forgetting 1.99→0.54 (~73%), new task kept; 4 tests |
| 8 | **Dual cognition** ✅ HARDENED | reflex (System 1) + deliberation (System 2), with compilation | `dual_process_engine.py`, `tests/test_dual_process.py` | every tier correct; deliberation compiles into instant memory; 7 tests |

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
- **#8 Dual cognition** ✅ — `dual_process_engine.py`: `DualProcessSolver` with
  three tiers (compiled memory -> policy reflex -> deliberation), composing the
  learned tree policy (reflex) and the hardened search (deliberation). Every
  tier's answer is correct on the spec; deliberated solutions compile into
  instant memory on recurrence. 7 tests.

### ✅ ALL 8 LAYERS HARDENED — the full hierarchy is production-validated.

Every capability is now a clean, tested module built on the hardened layer below
it, all green under `validate.py`. ~90 hardening tests; each rung a reversible
git tag (`hardened-knowledge` … `hardened-dual-cognition`).

## Enhancements — making dormant components load-bearing

Two components were measured as barely-helping (the binding memory was acting as
a slow dict; emotion was a weak learning-rate dial). These give them real jobs:

- **Semantic memory** (`semantic_memory.py`) — the binding memory with REAL
  embeddings (GloVe), so it does what a dict cannot: GENERALIZE via meaning.
  Learn a fact about "automobile", answer a query about "car" — never stored —
  because they mean nearly the same. Unrelated words correctly don't match
  (genuine similarity, not match-everything). 11 tests. Honest limit: only as
  good as the embeddings (GloVe-50 nails strong synonyms, noisy on subtle pairs).
- **Appraisal / redesigned emotion** (`appraisal_engine.py`) — grades input
  along pragmatic/affect dimensions from markers (surprise-weighted, so constant
  function words barely count), recognizing utterance TYPE (question / greeting /
  command) and tone (curious / friendly) from FORM before meaning. This is
  emotion's real job — a fast appraisal of the input — not the weak modulator it
  was. The tractable front of "understanding"; it does not break the meaning
  wall. 12 tests.

These are the input (appraisal) and memory (semantic) pieces of the
understand -> reason -> produce loop.

## 🔑 CAPSTONE — the understand -> reason -> produce loop

`conversation_engine.py` closes the loop, wiring the whole stack into controlled
conversation, fully symbolic and explainable:

- **understand** — AppraisalEngine (utterance type/tone) + intent recognition +
  working-memory context (resolves "it"/"that" to the current topic)
- **reason** — ReasoningEngine (facts, rules, transitive) — the hardened core
- **produce** — grammar-based verbalization of retrieved relations (article
  a/an, is/are agreement, pronoun for follow-ups) — GENERATED, not matched

```
teach: apple isa fruit / color red / grows_on tree / has seeds
> what is apple?   ->  An apple is a fruit. It is red. It grows on a tree. It has seeds.
> is it red?       ->  Yes, apple is red.        ("it" -> apple, from working memory)
> is it blue?      ->  Not that I know of.
> what is banana?  ->  I don't know anything about banana.
```

11 tests (greeting routing, multi-relation describe with article+pronoun,
confirm yes/no, coreference, honest unknown, topic switching, transitive). Every
word out is derived from a stored relation through a grammar rule — production,
not pattern-matching. Honest scope: controlled conversation; open-domain
comprehension remains the wall (needs an LLM).

## Learn by READING — the inverse of production

`fact_extractor.py` closes the read/write loop: the conversation engine turns
facts into sentences; this turns sentences back into facts. For controlled text
it parses (subject, relation, object) triples with grammar patterns — no LLM —
and resolves "it"/"they" to the running subject across sentences.

```
read:  "An apple is a fruit. It is red. It grows on a tree."
   ->  (apple, isa, fruit), (apple, is, red), (apple, grows_on, tree)
then:  "what is apple?"  ->  "An apple is a fruit. It is red. It grows on a tree."
```

So the brain can now learn by being FED TEXT, not only hand-typed facts — the
fuel for a real use. 12 tests (each relation type, coreference, topic switch,
read->learn->answer round-trip, validation). Pluggable: an LLM extractor for
messy/open text implements the same `extract()` interface and drops in (offline,
heavier) without changing anything downstream.

## The payoff lap — a verifiable expert it LEARNED BY READING

`domain_demo.py` runs the whole stack on a real (small) domain end to end: it
READS a family tree from plain text, is given two inheritance rules, and answers
multi-hop questions it was never told — with the derivation shown:

```
read:  "Tom is the parent of Sam. Sam is the parent of Kim. Kim is the parent of Ada."  ...
ask:   who is Tom's great-grandparent?
   ->  Ada
   because: tom parent sam AND sam grandparent ada => tom great_grandparent ada  [rule]
ask:   is Tom the parent of Ada?      -> no (Tom is Ada's great-grandparent)
ask:   who is Zara's grandparent?     -> I don't know anyone named Zara
```

text -> fact_extractor -> reasoning (facts + rules) -> answers + WHY, honest about
unknowns. The demonstration that everything built does something.

### Basic world knowledge — ConceptNet

`world_knowledge.py` + `world_demo.py` point the brain at real common-sense
knowledge: a curated English subset of ConceptNet 5.7 (~1500 high-weight
assertions about everyday concepts). The brain answers everyday questions over
it, deriving category membership through the FULL IsA taxonomy (multi-parent
closure) with the chain shown:

```
a dog can: bark, bite, chase ball     a car is: machine, vehicle
is a dog an animal?  -> yes   because: dog -> pet -> animal   (multi-hop, derived)
is a dog a vehicle?  -> not that I can derive
can a fish fly?      -> no (a fish can swim)
what is a zorblax?   -> I have never heard of a zorblax
```

Honest finding it surfaced, then FIXED: ConceptNet concepts have many parents, so
membership needs real transitive CLOSURE over the multi-valued fact graph — the
binding memory's associative recall returns one best parent and can't represent
it. First the demo hand-rolled a BFS; that closure is now PROMOTED into the
engine (`ReasoningEngine.closure` / `reaches` / `derive_all`), so the brain does
multi-parent reasoning natively and `world_demo` just calls it. The companion
fix `ask_all` explores ALL composition-rule bindings (two parents -> two
grandparents) instead of the first. This is the "feed it curated, trusted
knowledge" path (not crawl-the-web). Gate: 23/23.

### Talking about the world — world_chat

`world_chat.py` routes the same ConceptNet knowledge through the full
conversation loop, so the brain answers in GENERATED sentences, not print lines:

```
> what is a dog?       ->  A dog is a canine. It is an example of pet. ...
> is a dog an animal?  ->  Yes — dog -> pet -> animal.     (closure, chain shown)
> is a dog a vehicle?  ->  Not that I know of.
> what is a zorblax?   ->  I don't know anything about zorblax.
```

The conversation engine now routes category questions through the engine's
transitive closure (`reaches`) and all-bindings inference (`ask_all`), so the
two engine promotions above are what make "is a dog an animal?" answerable in a
sentence with the derivation shown. understand -> reason (multi-parent) ->
produce, over real world knowledge.

Produce stage now AGGREGATES: many objects of one relation collapse into a
single coordinated clause instead of one sentence each — a second grammar pass
over the structured relations (not a lossy re-parse of the generated text):

```
A dog is a canine, a pet and an animal. It can bark, bite and chase ball.
        (not: "A dog is a canine. It is a pet. It is an animal. It can bark. ...")
```

Fires only on multi-valued relations; one-object-per-relation output (apple) is
unchanged. `oxford()` joins "a, b and c"; isa/is share is/are agreement.

### Causal "how / why" — narrating a chain

`causal_demo.py` answers process questions. A process is a chain on one causal
relation (`leads_to`, `helps`): each step causes the next. Taught the steps, the
brain answers "how does X happen?" by walking the chain and verbalizing it —
backward to X's causes for "how does X grow/form", forward to X's effects
otherwise (`ReasoningEngine.process_chain`, a directional BFS; conversation adds
the how/why intent + `_narrate_chain`):

```
teach: sunlight leads_to photosynthesis leads_to sugar leads_to fruit leads_to apple
> how does an apple grow?   ->  Sunlight leads to photosynthesis, which leads to
                                sugar, which leads to fruit, which leads to apple.
> how does vitamin help?    ->  Vitamin helps immune system, which helps fighting
                                infection, which helps good health.
> how does a rock grow?     ->  I don't know how rock works.
```

No new reasoning power — a "how" question is a chain walk with a direction,
reusing the transitive core. Honest scope: linear taught processes, controlled
phrasing; it reorders and verbalizes given steps, doesn't discover the science.

## Workflow

1. **Prototype** a new capability (Python, fast iteration).
2. **Validate** it — add a check to `validate.py`, confirm the gate stays green.
3. **Promote** — once proven and stable, harden toward a real target / move into
   the C++ core for speed. One capability at a time; the scorecard + gate guard
   against regressions.

The C++ brain state persists via `save_components` / `load_components`
(round-trips cleanly). This file is the *capability* checkpoint — what the brain
can do and the evidence for it.
