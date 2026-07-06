# Brain v2 — Architecture Roadmap (neurosymbolic)

The design that came out of the proposer/executive design sessions. The thesis:
**the neural/LLM side reads and guides; the symbolic side composes and proves.**
Brain owns truth + reasoning + verified math; an LLM is a thin, shrinking
language shell. The defensible niche is *reliability, auditability, efficiency on
a bounded domain* — not general LLM parity (that needs LLM-scale weights).

## Status — 2026-07-02 (as-built; corrections dated, history not rewritten)
Much of this roadmap is now built. Current state is documented in
`brain2_architecture.md §11`. Highlights:
- **Three computing pillars live**: symbolic (verify), fuzzy (similarity), probabilistic
  (generation) — the probabilistic pillar (`prob_compute` → `neural_lm_torch`) was the
  missing third mode; language generation is now owned/internal.
- **7 verifier/regularity/proposer primitives ported to C++**, each verified == Python and
  locked in `harden_regress` (24/24); `harden_test` 35/35 no-crash. Guards run before ports.
- **Self-made verifiers** (`invariant_miner`, `verifier_monitor`, `check_library`,
  `synth_invariant`, `irregularity_detector`) and the **self-improving loop**
  (`conjecture_sandbox`, `autonomous_loop`).
- **Proposer** (once the weakest part) now learns online, transfers by task signature, and
  discovers its own features (`online_proposer`/`feature_learner`, ported to C++).
- **Training pipeline** (`train_pipeline.py`): qwen-coder teacher parses a domain into
  sentence⇒structure pairs — symbolic brain learns verified structure, student LM learns
  parsing — brain + student trained in one MPS pass. Teacher = bootstrap only.
- **Phase A slice implemented** (learned+verified templates `parse_template`/`template_memory`,
  `coverage_harness` LM-deletion metric, dimensional hard filter `domain_features`, concept
  naming/promotion `concept_memory`; test_phase_a 32/32). Skipped the plan's parts that
  duplicated existing tested modules (proposer trace/ranker, word grounder, front wiring).
- **Verifying agent** (`agent.py`): verify-each-step, flat success over horizon vs LLM decay.
- **Open-language track** (comprehension, built in dependency order; `test_open_lang` 29/29):
  event-frame logical form + its crisp membrane (`event_form`/`event_verify` — polarity
  non-contradiction + selectional type constraints, three-valued admit/reject/abstain, the
  contract built BEFORE the representation to de-risk store-flooding), discourse coref +
  typed connectives (`discourse`), autonomous reading loop with anti-collapse gate +
  per-fragment teacher escalation + measurable decay (`reading_loop`), and the honest
  taught-vs-wild coverage split (`coverage_harness.coverage_split`). Fluency stays owned-LM
  territory; this is where "open" comprehension is won, crisp all the way.

## The conservation law (why we build for our corner, not GPT's)
- LLM: max breadth + fluency, low reliability + efficiency + auditability.
- This architecture: max reliability + efficiency + auditability, low breadth + fluency.
- You trade the axes; you don't beat an LLM at being an LLM. Scaling deepens our
  corner (trust, domain mastery), it does not close the breadth/fluency gap.

## Load-bearing disciplines (rules, enforced everywhere)
- **The fuzzy/crisp membrane** — generalization never writes the truth store.
  Soft similarity may *suggest*; only the crisp store *answers*.
- **Verifier on everything** — proposer / induction / LLM all propose; only
  verified output ships. Honest "I don't know" on a miss.

## The unifying loop
```
[pattern-finder / neural]  conjectures a policy
        │
[verifier: inverse-check / proof / numeric gate]  admits only sound rules
        │
[means-ends executive, blackboard over policy+fact+working+episodic memory]
        │  uses verified policies + facts to solve, memoized
        ▼
   verified answer  →  LLM glorifies to English
```

## The components (from the design sessions)
**Exactness:** CAS breadth (SymPy) · search-based prover (one bounded domain) ·
verifier everywhere.
> **SymPy is a lever, not a permanent dependency.**  Right now SymPy does the symbolic
> heavy lifting (simplification, equation solving, algebraic manipulation) that the brain
> cannot yet perform on its own.  The correct end-state mirrors the `BRAIN2_GROUND_MATH`
> pattern: once the brain has grounded enough symbolic-reasoning procedures (the way it
> grounded arithmetic from succ/pred), SymPy should be *disabled* as a crutch and
> *reattached* only as a fast-path accelerator over the brain's own grounded reference.
> Flip the lever when: (1) the brain can solve the same class of expressions it currently
> delegates to SymPy, AND (2) results are verified on held-out inputs without SymPy in the
> path.  Until then, SymPy stays on — but always remember it is scaffolding, not structure.
**Language:** LLM as glorifier only · parser distilled FROM the LLM while it runs ·
two specialized shells (bigger encoder = understand, ultra-small decoder = fluency,
lean on GrammarMouth) · small purpose-built model for exactly the gap · definition-
grounded meaning (crisp core only; periphery stays the LLM; needs SOM grounding floor).
**Generalization:** a graph/embedding unit OUTSIDE the fact store.
**Reasoning/policy:** policies-in-memory · compose+verify · induction (feeds verifier) ·
**the Proposer** (similarity+success gated policy selector) · unified cross-subject
policy memory (typed, proposer-gated, unify on abstract structure) · cross-domain
signal gating (gate on STRUCTURE not domain label; adaptive aperture that opens when
stuck; keep it cheap).
**Memory/executive:** means-ends solver · all memories communicate not bind
(blackboard) · memoization/tabling.
**Product tiers:** lightweight (IoT, formal-in, no LLM) · middle (brain logic + small
LLM writes code → brain verifier checks) · research (build first).

**Rejected as stated:** transformer-that-collapses-to-a-tree (substrates don't
interconvert → use distillation) · engulf-LLM-weights-into-parser (= becoming the
LLM → instead the parser READS the frozen LLM's representations, verifier checks).

## Build order (cheapest-to-falsify first)
0. Verifier + memoization + the membrane rule. **[done in means_ends]**
1. ⭐ Proposer go/no-go — guided vs blind search. **PROVEN: 5.4× (program_synth_tree).**
2. Means-ends executive (blackboard over the memories). **[done: means_ends.py]**
3. Policies-in-memory + conjecture→verify. **[done: means_ends.py]**
4. Language: embedding-intent → distillation loop → two-shell.
5. Unified cross-subject policy memory + structural gating. *(after single-domain solid)*
6. CAS / prover / graph unit.

## Current status (artifacts)
- `means_ends.py` — Phase 2 executive + Phase 3 policy memory + conjecture/verify.
  Working & verified (derives values stored nowhere; rejects bad conjectures;
  persists learned policies).
- `program_synth_tree.py` — proposer concept proven at 5.4× (their existing bench).
- `proposer_experiment.py` — integration-as-search go/no-go. **Parked:** hand-rolled
  symbolic integration is the wrong substrate — the wrong by-parts branch blows up in
  EXPRESSION SIZE, not just node count, so it won't terminate cleanly. Needs SymPy
  (the CAS layer) to manage growth. The proposer concept is already proven elsewhere.

## Known latent bug — FIXED (2026-07-06)
- `physics_engine.ev` was flagged for eager op-dict eval (`{...,"^": a**b}[op]`) overflowing
  on `*`/`+`. Verified as-built: `ev` already evaluates lazily via an if/elif chain
  (`physics_engine.py:53-60`) — the overflow cannot occur. Note kept as resolved so external
  reviews don't re-flag it.

## Honest ceilings
- LLM-free math: formal input, covered domain → verified, sometimes hard. From English
  prose autonomously → no (autoformalization wall). Realistic: undergrad → qualifying-exam.
- Fed general facts: a precise auditable knowledge+reasoning engine (Wolfram-ish), NOT a
  chat LLM. Beats frontier LLMs on precision+honesty within coverage; far below on
  breadth/fluency. No single "GPT-X level" — pick the axis.
