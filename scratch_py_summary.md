### agent.py
```text
agent.py — the verifying agent: plan -> act -> VERIFY each step -> self-correct -> repeat.

Agentic tasks are multi-step, and that's where LLM agents fail: no step is verified, so
per-step error COMPOUNDS (0.95 per step -> 0.95^20 = 36% over 20 steps). This agent verifies
every step (stress-vs-oracle) before proceeding — a verified step doesn't compound error, and
an unverifiable one makes the agent ABSTAIN rather than march on with a mistake. So its
success stays flat with horizon length where an LLM agent's decays exponentially.

Each step: synthesize/compute an artifact, then stress-verify it. Fail -> self-correct (add
the counterexample, re-synth); still failing -> abstain. Reuses synth_engine (solve + stress)
and refute_synth (self-correction) — the organs already built.

    python3 agent.py
```

### algebra_engine.py
```text
algebra_engine.py — solve an equation for x, with the steps and a proof.

Generalizes the physics isolate() kernel from "rearrange a law" to "solve an
equation": given  left = right  with the unknown appearing once, isolate it,
evaluate, and VERIFY by substituting the answer back into the original equation
(so the solution is checked, not just produced).

    ae = AlgebraEngine()
    ae.solve(("=", ("+", ("*", 2, "x"), 3), 7))     # 2*x + 3 = 7  ->  x = 2

Honest scope: the unknown appears once (linear / single-power equations). Terms
on both sides, or x appearing twice, need term-collection — the next rung of the
search-based algebra (rewrite + solve), not this direct isolation.
```

### analogy_engine.py
```text
analogy_engine.py — structure mapping between two domains (bounded idea #2).

Analogy aligns RELATIONS, not surface features: a pump maps to a battery because
both *cause flow*, not because they resemble each other. Given two domains as
(subject, relation, object) facts that share a relation vocabulary, this finds an
object correspondence that preserves the relational structure, then TRANSFERS a
source fact whose analog is missing in the target — an analogical prediction.

    water:  pump increases flow,  pipe resists flow,  flow depends_on pressure,
            pump raises pressure
    elec:   battery increases current, resistor resists current,
            current depends_on voltage
    -> mapping pump:battery, flow:current, pipe:resistor, pressure:voltage
    -> predicts:  battery raises voltage     (the analog of 'pump raises pressure')

Honest scope: this is the realistic core of structure mapping — alignment by
relational signature over a SHARED vocabulary, with an unambiguous correspondence.
It is NOT open-ended invention (randomly mapping any domain onto any problem):
the mapping must be determined by the structure, and a transferred fact is a
HYPOTHESIS, not a truth — it should be verified (see inductive_engine) before it
is trusted. Ambiguous or structure-poor domains yield no mapping, honestly.
```

### analogy_struct.py
```text
analogy_struct.py — relational analogy WITHOUT a shared relation vocabulary (Novel #3).

The old analogy engine intersected pre-shared relation names (common = rels(A) & rels(B)):
it can only map domains that already use the same words. Real analogy (Rutherford: solar
system -> atom) maps domains whose relations have DIFFERENT names but the same STRUCTURE.

This is structure-mapping (Gentner): search for an entity alignment that makes the two
relational graphs correspond, inducing a RELATION mapping as it goes (pulls<->attracts,
revolves<->circles) — none of it pre-shared. Then it TRANSFERS: a source fact whose target
analog is missing becomes a predicted fact in the target domain (the point of analogy —
carry structure from the known domain to the new one).

Honest limit: brute-force over entity alignments (fine for small domains; the matching
problem is NP-hard in general). It maps relational STRUCTURE, not meaning — the transfer
is a hypothesis to verify, not a proven fact.
```

### appraisal_engine.py
```text
appraisal_engine.py — the input's pragmatic frame (and the redesigned emotion).

Before understanding WHAT an utterance means, recognize what KIND it is: a
question? a greeting? a command? and its tone — curious? friendly? Humans do
this from FORM ("what", inversion "are you", "?") before meaning. This grades
each word along pragmatic/affect dimensions, surprise-weighted so high-frequency
"constant" words ("are you") barely count and the informative markers carry the
signal.

This is emotion's real job — a fast, low-dimensional APPRAISAL of the input —
not the weak learning-rate modulator it was. It does NOT understand content
(that's the wall); it carves off the tractable pragmatic slice in front of it.

    AppraisalEngine().appraise("hey, how are you?")
      -> frame {greeting, friendly, question, curious, about_self}, type 'question'
```

### autonomous_loop.py
```text
autonomous_loop.py — the brain drives itself: curiosity -> conjecture -> test -> bank -> learn.

Every piece existed separately. This wires them into ONE standing, self-improving cycle that
runs with no human stating laws or goals:

  curiosity  : pick an unknown quantity (a gap) to explain
  conjecture : the PROPOSER offers candidate forms, ordered by what it has learned
  test       : the SANDBOX checks each against a trusted principle (no answer key handed in)
  admit/bank : the survivor becomes a verified law, stored
  learn      : the winning form's shape raises the proposer's prior -> next gap is cheaper

The self-improvement is MEASURED: gaps that share structure are solved with FEWER conjectures
as the loop runs, because the proposer learned the shape from earlier gaps. That is the
toolbox becoming a system — the parts compound instead of sitting as demos.

Honest limit: conjectures are drawn from a form grammar (compositions of known ops); a truly
unprecedented structure outside the grammar is still irreducible search. The loop makes the
brain better at novel-but-related, not at zero-precedent novelty.
```

### brain_chat.py
```text
brain_chat.py — one front door to every faculty.

A thin router: detect the KIND of input and dispatch to the engine that handles
it. Math notation is a formal grammar (route to the exact math engines); anything
else is natural language (route to the controlled conversation engine, which
itself handles facts, how/why chains, and arithmetic word problems). The router
holds no intelligence — it only decides who answers.

    bc = BrainChat()
    bc.learn("dog", "isa", "animal"); bc.set_transitive("isa")
    bc.respond("differentiate sin(x^2)")   -> calculus
    bc.respond("is a dog an animal?")      -> reasoning
    bc.respond("I have 10 apples ...")     -> word math

Honest scope: dispatch over the existing engines' scopes — no new reasoning, just
a unified entry point. Math is exact; natural language stays controlled.
```

### brain_codegen.py
```text
brain_codegen.py — the brain builds the LOGIC; rendering is mechanical (no LLM).

The sharper middle-tier thesis: don't let the LLM think. The brain DISCOVERS the
logic (guided induction over examples -> a verified formula IR), and that IR is
rendered to Python deterministically. The LLM does nothing — there's no algorithm
for it to get wrong, because the brain already found and VERIFIED it.

  examples -> guided induction -> verified formula IR -> mechanical render -> code
                                                       -> re-verify generated code

For formula-shaped logic this needs no LLM at all. (For control-flow logic — loops,
recursion — the brain would emit a plan and an LLM would TRANSCRIBE it; the LLM is a
transcriber, never the thinker, and the test gate still catches mis-transcription.)

    python3 brain_codegen.py
```

### brain_data.py
```text
brain_data.py — ingest the tagged training corpus (see docs/kimi_data_prompt.txt) and feed
every subsystem from ONE file. The bidirectional-template insight as a data pipeline:

  FACT / LAW  -> the symbolic brain (knowledge_distill: facts taught, laws verified)
              -> the student LM corpus (sentence => structure parse pairs)
  EVENT       -> event pairs (parser target + the mouth's say_event target)
              -> the predictor (ordered events -> learned verb transitions)
  ISA         -> the type oracle (grounds selectional types)
  MORPH       -> the mouth's morphology table (child-grade -> fluent, by LEARNING not an LLM)
  SEQ         -> the predictive loop (coherent event streams) + raw text for the C++ Brain

Line grammars (one per line):
  <sentence> => FACT: obj | prop | num
  <sentence> => LAW: quantity = expression
  <sentence> => EVENT: verb | agent | patient | tense | polarity(+/-)
  ISA: child | parent
  MORPH: lemma | past | present3sg
  SEQ: <sentence>
```

### brain_planner.py
```text
brain_planner.py — the two engines in one loop.

Binding memory KNOWS things (facts taught online). Tree search REASONS to a
goal. This joins them: the brain stores a world model as facts, and the search
plans over that model by *querying the brain* — then explains the plan.

The world is taught as facts:
    smelt requires ore     smelt produces iron
    chop  requires axe     chop  produces wood
    forge requires iron    forge requires wood    forge produces sword

Then: "you have ore and axe, get a sword." The planner asks the brain what each
action needs and makes (via query_all), searches for an order that works, and
reads back the plan. This is genuine multi-precondition planning — forge needs
iron AND wood, which come from two different branches — so it is strictly more
than transitive closure.

Honest about the join: facts live in the real binding memory and are retrieved
through the brain's query_all; the search is tree_reason's general engine. The
brain is the knowledge; the search is the reasoning over it.
```

### brain_repl.py
```text
brain_repl.py — teach it facts, ask it something you never stated.

The demo for what brain2 actually is: a CPU-native system that learns facts
online (no retraining), derives conclusions it was never told, and explains
its reasoning by reading back the actual chain — because the reasoning is an
explicit traversal of its binding memory, not hidden weights.

Usage:
    python3 brain_repl.py            # interactive
    python3 brain_repl.py --demo     # scripted walkthrough

Interactive grammar (entities and relations are single tokens):
    alice > bob            teach a fact            (alice  >  bob)
    tom parent sam         teach with any relation
    alice > ?              ask: what does alice relate to (transitively)?
    alice > emma           ask yes/no, with derivation
    facts                  list everything taught
    help / quit
```

### brain_session.py
```text
brain_session.py — the operational shell: a bootable, ingestable, queryable brain.

Ties the product spine into one running system. A persistent KnowledgeBase loads
into the Brain at boot; the Mind answers via eyes -> brain -> mouth; you can ingest
more knowledge live and watch coverage grow mid-session. The REPL is thin glue —
the logic lives in BrainSession (so it is testable).

    sess = BrainSession()
    sess.boot_conceptnet()
    sess.ask("what is a dog?")
    sess.ingest_text("A whale is a mammal. It lives in the ocean.")
    sess.ask("what is a whale?")

CLI:
    python3 brain_session.py [kb.json]        # boot from a saved KB (or ConceptNet)
    :ingest <file>   :stats   :coverage   :save <file>   :quit
```

### brain_store.py
```text
brain_store.py — the brain ACCUMULATES verified knowledge across sessions.

Until now every demo re-derived everything from scratch and forgot it on exit. This
is the self-extension made permanent: a persistent store of what the brain has
LEARNED — discovered formulas (policies), facts, and synthesized functions — each
gated by verification before it's admitted, saved to disk, and reloaded next run. The
brain never re-derives what it already knows; it builds on top.

  session: load store -> learn only the NEW things (verify -> admit) -> save
  across sessions the store GROWS, and prior knowledge is reused immediately.

    python3 brain_store.py        # runs two sessions, shows the store growing
```

### calculus_engine.py
```text
calculus_engine.py — symbolic differentiation: many rules COMPOSED in one pass.

This is the concrete answer to "can the brain switch between policies?" Each
differentiation rule (power, product, quotient, chain, trig, exp, log) is a
policy. A single expression forces several to compose — d/dx(sin(x^2)) needs the
chain rule, the trig rule, and the power rule together — and the engine applies
exactly the ones the expression demands, reporting which fired.

Differentiation is mechanical and complete, so the result is correct BY
CONSTRUCTION (no search, no guessing). Expressions are nested tuples:

    ("^", "x", 3)            x^3
    ("*", ("^","x",2), ("sin","x"))     x^2 * sin(x)
    ("sin", ("^","x",2))     sin(x^2)

    CalculusEngine().diff(expr) -> Result(expr, simplified_str, rules_used)

Honest scope: differentiation (a finite, mechanical ruleset). Integration is the
SEARCH case (rules can apply many ways, not mechanical) — a separate build.
```

### check_library.py
```text
check_library.py — a self-growing, PERSISTED library of verifiers (refuter -> invariant -> store).

Connects three pieces into one loop that makes the envelope grow across sessions:

  refuter finds a candidate BREAKS  ->  mine the invariants the CORRECT version satisfies
  ->  bank them in a persistent library (keyed by task)  ->  next session they pre-filter
      candidates immediately, no re-mining.

So every overfit the brain catches leaves behind a cheap, reusable check. The library is to
verifiers what BrainStore is to facts/functions: self-extension made permanent. Over sessions
the check set GROWS and prior checks are reused for free.

Honest scope: invariants are necessary-not-sufficient (from invariant_miner) — banked checks
reject cheaply, never falsely accept. Library is per-task-key; cross-task generalization of a
check is a further step.

    python3 check_library.py     # two sessions: session 1 learns from a break, session 2 reuses
```

### code_gen.py
```text
code_gen.py — generate class boilerplate from a structured spec (Py / C++ / Java).

The honest reading of "build something from proper instructions": a structured
spec (class name, typed fields, method signatures) is PRODUCED into idiomatic
code per language — the same grammar-production idea as the sentence generator,
applied to code. Deterministic and correct by construction for the supported
constructs; it is not logic synthesis and does not write method bodies.

    spec = ClassSpec("Point", [Field("x","int"), Field("y","int")],
                     [Method("distance", [], "float")])
    CodeGenerator().generate(spec, "python")   # -> a valid Python class

Honest scope: OOP scaffolding (fields, constructor, method stubs) across the
three languages + their type mappings. Method LOGIC is left as a stub — writing
arbitrary algorithm bodies is the LLM/synthesis frontier, not template production.
```

### component_validation.py
```text
component_validation.py — Step 3, done as validation, not deletion.

Each brain component claims to do a job. This suite measures whether it does,
ON THE AXIS WHERE IT IS SUPPOSED TO ACT — not on perplexity, which is the wrong
ruler for dreaming or emotion (the same mistake as judging reasoning by
perplexity). A component that proves its effect is kept WITH A NUMBER behind
it; one that shows nothing is first a wiring bug to investigate, not a delete.

Tests:
  1. dream_consolidation — does the dream cycle's episodic replay protect
     recently-learned material from interference by later learning?
       protocol: learn A -> (dream A | skip) -> learn B (interference)
                 -> measure retention NLL on held-out A.  dreaming should
                 lower A's loss after B.
  2. emotion_modulation  — does emotion's learning-rate modulation make
     surprising/salient items better retained?  (requires the emotion-disable
     knob; see emotion_enabled.)
```

### composable_proposer.py
```text
composable_proposer.py — a learned PROPOSER guides compositional code synthesis.

composable_synth brute-enumerates the composition space. That explodes as the DSL
grows — exactly the warning from cross-domain policies. The fix is the same as
program_synth_tree: a learned proposer that, from FEATURES of the target's I/O,
predicts which primitive belongs in each slot, so the search evaluates the likely
programs first and finds the answer in far fewer tries.

  train: random programs -> (I/O features, slot choices)  -> a decision tree / slot
  solve: features(target) -> per-slot choice distributions -> order compositions by
         likelihood -> first exact fit. Count programs evaluated, blind vs guided.

The brain proposes WHICH pieces to compose (premise selection); the verifier still
confirms the survivor. This is what makes composable synthesis scale.

    python3 composable_proposer.py
```

### composable_synth.py
```text
composable_synth.py — coding primitives as COMPOSABLE policies (cross-class novelty).

The same move as cross-domain policy composition, applied to code. Instead of N
separate templates (fold, two-state, conditional, early-return), there is ONE
composable program shape whose PARTS recombine:

  a, b = INIT
  for i in RANGE:
      if GUARD(i, n):              # optional
          a, b = UA(a,b,i), UB(a,b,i)
      if EARLY(a,b,i,n):           # optional
          return i
  return FINAL

Searching combinations of {init, range, guard, update, early, final} gives:
  * the OLD classes as compositions (Fibonacci = two-state; count-divisors = guarded
    fold) — combinatorial coverage, not hand-built templates, AND
  * NOVEL cross-class algorithms no single template had — e.g. a FOLD combined with
    an EARLY-RETURN ("first k where 1+..+k >= n"), which is neither a pure fold nor a
    pure search.

Every candidate is verified on held-out examples. (At scale this space explodes —
that's where the PROPOSER gates which pieces to compose, exactly like the policy
proposer. Bounded here to demonstrate the principle.)

    python3 composable_synth.py
```

### compositional.py
```text
compositional.py — systematic generalization: novel combinations of known parts.

The thing pure pattern-matchers (and LLMs, on SCAN/COGS) fail and the symbolic
core gets for free. Give the brain only ATOMIC policies (each one hop) and facts
for a FRESH entity it never saw. It answers DEEP targets by composing those atoms
into chains never trained as a unit — force -> work -> energy -> power, four levels
deep, assembled on demand.

  K atomic policies  ->  answers targets at every reachable depth, on any entity,
  via recombination. Coverage is COMBINATORIAL in the atoms, not enumerated.

A student/LLM would need each combination in its training data; the executive
derives them, because meaning here is COMPOSITIONAL by construction.

    venv2/bin/python3 compositional.py
```

### concept_blend.py
```text
concept_blend.py — the novelty primitive: make a concept that exists in NEITHER parent.

Analogy/transfer finds the nearest EXISTING concept (generalization). Blending does the
opposite: deliberately fuse two DISTANT concepts into a point that lands in EMPTY feature
space — outside every known category. That empty point is a candidate new solution space
(the thing the architecture otherwise structurally lacks). A blend is only interesting if
it is verifiably NOVEL: its nearest known concept is farther than the cluster radius, so
it is classified as neither parent nor anything else.

Pure-python (feature vectors as lists) so it runs anywhere. In the real brain these are
SOM centroids; a verified-novel blend would mint a new SOM neuron at that point and train
it on the combined features.

Honest limit: blending PROPOSES novelty (an empty, reachable region). It does not prove
the new concept is USEFUL — that still needs grounding/verification downstream. It widens
the space; the verifier still decides what's real.
```

### concept_memory.py
```text
concept_memory.py — from shared structure to NAMED, REUSABLE concept.

factorizer/curiosity_cross discover that two domains share a shape; this store gives the
shape a name and a life-cycle: candidate -> (used PROMOTE_AT times in verified solutions) ->
promoted. Promotion is the statistical admit gate — a concept earns first-class status by
proving reusable, not by being found once. A promoted concept is a hypothesis with a good
track record, NOT a truth: it still goes through verification every time it's used; promotion
changes what the proposer PROPOSES, never what the verifier ACCEPTS. Shape variables are
UPPERCASE strings; recognize() pattern-matches a concrete expr and returns the binding.
(Plan Phase A, Task 11 — the naming/promotion lifecycle factorizer was missing.)
```

### conceptnet_taxonomy.py
```text
conceptnet_taxonomy.py — relation-FILTERED ConceptNet ingest for real reasoning.

scale_test streamed the first-N edges (alphabetical wiktionary antonyms) -> almost no
IsA -> closure found 0 ancestors. That was a sampling artifact, not a reasoning limit.
This filters the stream to TAXONOMY + PROPERTY relations (IsA, PartOf, HasA,
CapableOf, UsedFor, HasProperty, MadeOf, AtLocation), so the loaded graph is a real
ontology with multi-hop chains — then reasons over it at scale: transitive IsA
closure, property inheritance, and which concepts share an ancestor.

    python3 conceptnet_taxonomy.py
```

### conjecture_sandbox.py
```text
conjecture_sandbox.py — the brain TESTS its own guesses (active experimentation).

So far verification was passive: check a candidate against data it was given. This lets the
brain ACT on an unverified guess it feels confident enough to test — design an experiment,
run it in a sandbox, and judge the conjecture against the knowledge it ALREADY TRUSTS. This
is how a scientist promotes a hunch: not by being told, but by testing it.

  conjecture (an unverified formula/rule)
     -> DESIGN a test: generate diverse scenarios spanning the input space
     -> RUN it in the sandbox and derive what a TRUSTED principle says the answer must be
     -> JUDGE: survives every self-generated test -> PROVISIONAL ADMIT (confidence up);
        contradicted -> REJECT with the counterexample.

The oracle is the brain's own VERIFIED knowledge (here: energy conservation). No external
answer key, no human — the brain bootstraps new beliefs from trusted ones by experiment.

Honest limit: it can only test a guess against principles it ALREADY trusts (or internal
consistency). A guess with NO trusted anchor and no consistency handle stays unverifiable —
the sandbox extends reach where a trusted principle can adjudicate, not everywhere.
```

### context_embed.py
```text
context_embed.py — word meaning from CONTEXT, not a hand table (toward open comprehension).

The parser only knew words it was TOLD (plus a hand synonym map). Open comprehension needs
meaning to come from context, the way grounding derives perceptual concepts from data
instead of being told. This builds count-based distributional vectors: a word's meaning is
the company it keeps (co-occurrence over a corpus). Then an unseen/paraphrase word maps to
the nearest KNOWN canonical word by contextual similarity — no synonym table, no training.

This is the FUZZY proposer half of the membrane: context gives a rough reading
(velocity ~ speed), the crisp reasoner still verifies the structured query it proposes.
A paraphrase routes only if its meaning lands on something checkable.

Honest limit: count-based vectors need a corpus and give SIMILARITY, not precise meaning;
quality scales with corpus size. Real open comprehension needs far more text. This is the
lightweight proof that meaning-from-context generalizes past the hand table.
```

### conversation_engine.py
```text
conversation_engine.py — the understand -> reason -> produce loop (capstone).

Ties the whole stack together for CONTROLLED conversation, fully symbolic and
explainable:

  understand : AppraisalEngine (utterance type/tone) + intent recognition +
               working-memory context (resolves "it" / "that" to the topic)
  reason     : ReasoningEngine (facts, rules, transitive) — the hardened core
  produce    : grammar-based verbalization of the retrieved relations
               (articles a/an, is/are agreement) — generated, not pattern-matched

    c = ConversationEngine()
    c.learn("apple", "isa", "fruit"); c.learn("apple", "color", "red")
    c.respond("what is apple?")  -> "An apple is a fruit. It is red."
    c.respond("is it red?")      -> "Yes."        ("it" -> apple, from context)

Honest scope: controlled conversation. Intent recognition is form-based over a
defined set of question shapes; genuine open-domain comprehension is the wall
(that needs an LLM). Within the controlled set, every word out is derived from a
stored relation through a grammar rule.
```

### core_knowledge.py
```text
core_knowledge.py — a small, hand-verified seed of high-quality world facts.

"Super quality data" honestly means CURATED and VETTED, not scraped. These are
clean (subject, relation, object) triples a person can check by eye, spanning a
few everyday domains, so the brain has a trustworthy base to reason over and to
chain (isa is transitive). Scale comes later from ConceptNet / Wikidata via
`knowledge_base`; this is the gold core.

    from core_knowledge import CORE_FACTS, load_core
    load_core(knowledge_base)        # ingest the vetted seed
```

### corpus_scale.py
```text
corpus_scale.py — pull the lever: more corpus -> more lexical generalization, MEASURED.

context_embed's mechanism (meaning from co-occurrence) scales with text. This measures it:
the SAME paraphrase test set, scored against a small corpus vs a larger one. Coverage —
how many paraphrases map to the right canonical concept — should jump with more text, with
no code change. That is the whole point of meaning-from-context over a hand table: the
ceiling is "how much text", not "how many words a human typed".

Honest: still a synthetic corpus (no web access here); the MEASUREMENT is the real result —
coverage rises with corpus size. Real deployment points this at a large natural corpus.
```

### coverage_harness.py
```text
coverage_harness.py — the deletion metric. Measures which rung resolves each held-out
question and whether the resolved rel is CORRECT. A student/LLM rung is deletable for a
domain when template_pct clears threshold on FROZEN held-out data — not by vibes.
(Plan Phase A, Task 7.)
```

### cpp_accel.py
```text
cpp_accel.py — optional C++ fast paths (the compiled brain2 .so) with a Python fallback.

The core primitives were ported to C++ and PROVEN equal to their Python reference in
harden_regress (brain2.<fn> == <fn>_py). This module is where those verified ports get wired
into the RUNTIME: a caller uses the C++ path when brain2 is importable, else the identical
Python path. Guarded, so a pure-Python environment (no compiled brain2) still runs.

Honest per-port assessment (why NOT all 9 are wired — a port being verified-equal does not make
it a safe drop-in):

  WIRED (clean signature, verified ==, and worth it):
    * law_error       — least-squares fit; float->float, not in a hot loop. Real compute.

  DELIBERATELY NOT WIRED (documented, not oversight):
    * cosine_map      — clean, but called per word-pair in tight vocab loops; pybind
                        marshalling overhead would SLOW it. Native Python wins at this size.
    * disc_weights /  — couples (weights feed feat_sim); must swap as a pair, trivial compute,
      feat_sim          no measurable gain. Left Python to avoid format-coupling risk.
    * inv_mine        — C++ is a SUBSET (lacks 'monotonic_increasing'); swapping would DROP a
                        real invariant -> behaviour regression. harden_regress line ~90 shows it.
    * refute_int1     — takes precomputed candidate/oracle arrays, not refute()'s (f, oracle)
                        shape; and has an honest 64-bit-output limit. Needs a refactor to wire.
    * eval_sexpr      — takes a SERIALIZED s-expression string; converting tree->string per call
                        would be slower than native tree eval in hot factorizer loops.
    * analogy_score   — returns only the score; callers (align/align_greedy) also need the
                        relation map (relmap) the Python _score returns. Partial output.

The real value of the ports is a correctness-proven C++ implementation READY for when data
scales (where marshalling overhead is amortized) — not a current speedup at these input sizes.
```

### crispify_bridge.py
```text
crispify_bridge.py — feed the C++ PolicyEngine from the C++ BindingMemory.

The membrane, made concrete at the vector seam. The C++ BindingMemory is FUZZY
(stores triples as vectors, recalls by similarity with a confidence). The C++
PolicyEngine is CRISP (exact arithmetic over verified facts). The bridge:

  query BindingMemory  ->  (object_vector, confidence)  ->  GATE on confidence
     accept (conf >= floor): decode the scalar -> a crisp fact for the engine
     reject (low conf):      return None -> honest "unknown"

So fuzzy associative recall PROPOSES, the confidence gate DISPOSES, and only crisp,
high-confidence facts reach the reasoner. Verified empirically: exact bound facts
recall at conf 1.0; an unstored fact recalls at ~0.6 and is rejected. Both halves
are C++; this is the only Python — the encode/decode + the gate.

Run:  venv2/bin/python3 crispify_bridge.py
```

### curiosity_cross.py
```text
curiosity_cross.py — curiosity into the ADJACENT POSSIBLE, not just data gaps (Novel #2).

The existing curiosity loop fills WITHIN-domain gaps: it promotes rules the data already
supports ("cannot invent a pattern that isn't there"). It misses the case where two
domains, each unremarkable alone, share an abstract structure that is only an insight when
you put them side by side.

This is that move, using the factorizer's anti-unification ACROSS domains: factor the
UNION of two domain libraries; a primitive that appears in BOTH is a cross-domain law —
the adjacent-possible insight neither domain shows on its own. Then it transfers: the
shared structure predicts the form of a quantity in one domain from the other.

Honest limit: still verified (the shared structure must actually generalize both
formulas); it discovers shared STRUCTURE, not brand-new physics. But "these two distant
things are the same shape" is exactly the cross-domain spark within-domain curiosity can't reach.
```

### curiosity_loop.py
```text
curiosity_loop.py — idle-time learning driven by prediction error (idea #4).

When not being asked anything, the brain explores its own knowledge gaps. It
predicts what follows each observed event; where it is wrong (or has no rule), the
prediction error is high — that is curiosity. Each idle tick it accumulates more
observations, mines + verifies new rules (the inductive engine) to cover the
high-error areas, and its error drops. Attention then moves to whatever is still
unpredictable.

    cl = CuriosityLoop()
    cl.run([chunk1, chunk2, ...])      # idle ticks, each folds in more data

Honest scope: curiosity = prediction error; exploration = the verify-before-
promote inductive loop. It learns patterns that REPLICATE and stays curious where
none exist (a random follower never yields a verified rule, so the gap persists —
the loop cannot invent a pattern that isn't there). It does not discover new math;
it fills gaps with rules the data actually supports.
```

### deeper_grammar.py
```text
deeper_grammar.py — boolean / multi-clause conditions (the next structural rung).

nested_parser handled a single if/then condition. This adds compound boolean conditions:
"if A and B then C", "if A or B then C", with each atom a verified comparison and the
entity carried across clauses (pronoun-style). Each atom is solved by the crisp engine, so
the whole boolean is verified end to end; an unparseable/unverifiable atom -> abstain.

  atom      : "<entity?> <relation> (greater|less) than <number>"
  condition : atom (and|or atom)*   evaluated left-to-right within one operator
  rule      : "if <condition> then <query>"

Honest limit: one boolean operator per condition (all-and or all-or), comparisons against
numeric literals. Mixed and/or precedence and clause-vs-clause comparisons are the next rung.
```

### dimensional_verify.py
```text
dimensional_verify.py — a new VERIFIER: dimensional analysis.

The architecture's reach = the reach of its verifiers. This adds one: units. A
formula must be DIMENSIONALLY consistent (force is kg·m/s², not kg+m/s²), checked
by unit algebra over base dimensions [Mass, Length, Time]. It's a SECOND,
independent gate on induced policies: a formula that fits the data numerically but
is dimensional nonsense (a coincidental overfit) gets rejected — without running a
single extra data point.

  units("mass*accel") = (1,1,-2) = force  -> consistent
  units("mass+accel") = mismatch          -> INVALID (can't add kg and m/s²)
  units("mass*speed") = (1,1,-1) = momentum != force -> rejected for 'force'

Pairs with policy_induction: induce by fit, then KEEP only if dimensionally sound.

    python3 dimensional_verify.py
```

### discourse.py
```text
discourse.py — the jump from sentence to paragraph.

Three cheap, crisp mechanisms; no learning yet — markers-first, like the earliest templates:

  * Coref     — a pronoun resolves to the most recent TYPE-COMPATIBLE entity on the context
                stack. Pure pointer resolution over working memory: symbolic and cheap.
  * Connectives — because/but/so/then become typed Relations (CAUSE/CONTRAST/SEQUENCE)
                between the surrounding event ids. Explicit markers only; implicit discourse
                stays unjudged (abstain, not guess).
  * ContextStack — the entities/events seen so far, persisting across sentences/turns = the
                dialogue state whole_brain wires in.

(Open-language track, Gap 3.)
```

### domain_features.py
```text
domain_features.py — domain knowledge the proposer can use.

Hard filter: dimensional analysis. A dimensionally-inconsistent policy cannot be correct, so
it is pruned BEFORE search (score 0) — pure structure, zero learning, the strongest pruning
signal physics/chem offer. Unknown units -> None (abstain): the filter must NEVER prune what
it does not understand (the three-valued True/False/None contract). Soft feature: per-policy
Laplace-smoothed success rate. (Plan Phase A, Task 10.)
```

### dp_greedy_synth.py
```text
dp_greedy_synth.py — DP + greedy synthesis, gated by STRESS vs a brute-force oracle.

Toward "the brain writes the exotic algorithms too" — honestly. Two truths:

  1. The brain CAN search DP/greedy templates and find the recurrence/strategy.
  2. Fitting a few examples is NOT proof. DP recurrences overfit; greedy rules that
     work on small cases are often WRONG in general. So the verifier here is
     stronger: run the candidate against a BRUTE-FORCE oracle on hundreds of random
     inputs. Admit only what matches everywhere tried; REJECT greedy that's a trap.

  DP   (max subarray): search cur/best updates -> stress vs brute -> finds Kadane.
  GREEDY (coin change): fixed greedy strategy -> stress vs DP-optimal oracle ->
         ADMITTED for canonical coins, REJECTED for {1,3,4} (greedy isn't optimal).

The point: the brain proposes the clever algorithm; the brute-force oracle decides
whether it's actually correct — so a synthesized "exotic" algorithm is trustworthy
only when it survives the stronger gate, and the gate honestly fails the traps.

    python3 dp_greedy_synth.py
```

### dp_proposer.py
```text
dp_proposer.py — learned proposer over the DP recurrence space.

dp_greedy_synth searched a tiny DP space (Kadane only). This expands it to a real
recurrence DSL (init x cur-update x best-update ~ 56 recurrences spanning max-subarray,
min-subarray, max-element, sum) and puts a learned PROPOSER on top: from features of
the task's I/O it predicts the recurrence pieces, ordering the search so the right
recurrence is found in far fewer tries. Gate = held-out examples (plus stress-vs-brute
where an oracle exists, e.g. Kadane).

Measured vs a FAIR random-order baseline (fixed enumeration order rigs it).

    python3 dp_proposer.py
```

### dual_process.py
```text
dual_process.py — reflex (System 1) + deliberation (System 2).

The HFT objection was that reasoning is deliberative (slow search) while a
reflex is instant. The answer is to have BOTH, like a mind does: a fast
reflex that acts without searching, and slow deliberation that kicks in only
when the reflex is unsure.

Here the learned tree policy IS the reflex: follow its top choice at each step
with NO search (a greedy rollout — a handful of cheap policy lookups). If that
solves the task, done, instantly. If not, fall back to full tree search
(deliberation). On a stream of mixed-difficulty tasks the reflex handles the
familiar/easy ones for ~free, and deliberation is spent only on the genuinely
hard ones — so average effort collapses.

This is how expertise works: practiced situations become intuition (reflex),
novel ones still need thought (deliberation). The compilation bridge — caching
a deliberated solution as a future reflex — is the next step (the brain already
has procedural memory for exactly this).
```

### dual_process_engine.py
```text
dual_process_engine.py — hardened Dual cognition (milestone #8, the last rung).

Reflex (System 1) + deliberation (System 2) + compilation, as a clean solver.
Three tiers, fastest first:
  1. compiled memory — a cache of already-solved tasks (instant)
  2. policy reflex   — greedy rollout of the learned tree policy, NO search
  3. deliberation    — full search for the genuinely novel
and every deliberated (or reflex) solution is COMPILED into the cache, so a
recurring task is answered instantly next time.

    s = DualProcessSolver(train_policy())
    r = s.solve(examples)        # r.tier in {memory, reflex, deliberation}
    r.found, r.apply(new_input)

Composes hardened pieces: the tree policy (reflex) and the search engine
(deliberation). Whatever tier answers, the program is correct on the examples.
```

### event_form.py
```text
event_form.py — the richer logical form that holds open language.

FACT: obj|prop|num and LAW: target=expr are too shallow for prose — no slot for negation,
causality, agent/patient, tense. This is the extension (not replacement): an Event carries
verb + roles + time + polarity, and a Relation typed-links two events (CAUSE/CONTRAST/
SEQUENCE). FACT/LAW remain degenerate cases (a stative Event with patient=value), so
everything downstream re-targets to Events without throwing away the numeric core.

Membrane note: an Event is a CONJECTURE until event_verify admits it. This module only
builds/serializes the shape — it owns no truth. (Open-language track, Gap 1.)
```

### event_parse.py
```text
event_parse.py — the INTAKE the event membrane was waiting for.

Until now templates only emitted stative FACTs (unconstrained verbs), so event_verify's real
work — constrained verbs, negation, tense, causality — never fired on prose. This turns a
sentence into an Event(verb, agent, patient, time, polarity):

  * negation  — not / never / -n't  -> polarity NEG (the claim is denied)
  * tense     — did/was/-ed/irregular-past -> past; will/shall -> future; else present
  * SVO       — agent = nearest entity/pronoun before the verb, patient = nearest after

Markers-first and crisp (same stage as discourse.py) — no learning yet; a bad parse yields
None (abstain), it never guesses an Event into existence. Pronouns stay as tokens for the
reader's coref to resolve. verbs is the known-lemma set (crisp verb identification).
(Open-language track — closes the intake gap: now prose reaches the membrane.)
```

### event_predict.py
```text
event_predict.py — predictive processing for the event stream (the default-state upgrade).

Brains don't idle waiting for input; the cortex CONSTANTLY predicts the next input and learns
from the error (Friston/Clark predictive processing). The reactive front only answers when
asked. This bolts prediction onto the event intake:

    predict next event  ->  parse the real one  ->  SURPRISE = prediction error  ->  learn

Surprise is the real, SEMANTIC novelty signal (an *unexpected event*), not the shallow lexical
novelty (an unseen token) the front used before. It drives what a brain does with error:
  * high surprise -> worth storing (episodic), worth attention
  * where you're wrong -> where to learn (update the priors)
  * high-error regions -> curiosity targets (the gaps to read toward)

Membrane: prediction is a fuzzy expectation — it never asserts truth. The parse + crisp
membrane still own what actually happened. Learning is Hebbian in spirit: consecutive events
that co-occur strengthen the transition, so the brain comes to expect what usually follows.
```

### event_verify.py
```text
event_verify.py — the membrane for events. The numeric gate does not apply to prose, so
events get a WEAKER but still crisp contract:

  1. Polarity non-contradiction — the store may not hold EVENT(v,a,p,t,+) and (v,a,p,t,-).
  2. Selectional type constraints — a verb restricts the type of its agent/patient
     (type_of maps a token to a type via SOM clusters / semantic memory, injected).

Three-valued, exactly like domain_features' dimensional filter: admit / reject / abstain.
ABSTAIN is the load-bearing case — held, never guessed — in TWO situations:
  (a) a CONSTRAINED verb whose role type we don't yet know, and
  (b) a verb we have NEVER SEEN (open-world closed assumption): with no corpus evidence
      the verb even exists, we cannot vouch for it, so we hold rather than admit.
An UNCONSTRAINED but KNOWN verb has no selectional claim to check, so a numeric-core fact
flows straight through. Distinguishing (b) from a known-unconstrained verb requires the
vocabulary — pass `known_verbs`; omit it (None) to keep the legacy "unconstrained -> admit"
behavior. Abstained events do not enter the truth store and do not contradict; they are the
escalation queue for the reading loop.

Fuzzy proposes an Event; this disposes. (Open-language track, Gap 1 — contract before scale.)
```

### exam.py
```text
exam.py — Three-level curriculum exam for brain2 (grades 1-8 training data).

EASY   (100): Direct fact/prop recall, basic ISA, known-verb parse, morph
MEDIUM (100): Multi-hop ISA, membrane decisions, cross-grade facts, predictor
HARD   (100): Novel entities, abstain discipline, type-grounded inference, contradiction tolerance

Run:  /opt/homebrew/bin/python3.13 exam.py
```

### exam_math.py
```text
exam_math.py — does the brain COMPUTE the curriculum's arithmetic?

The critique was right: the old exam tested zero math. The curriculum's math
files are full of arithmetic identities — "LAW: 9 + 1 = 10", "LAW: 6 * 3 = 18",
"LAW: 20 - 14 = 6" — that KD dropped (their LHS isn't a variable name, so they
were never admitted as policies). But they ARE checkable computations.

This exam pulls every such identity from math1-8, evaluates the LEFT side with
the brain's LEARNED arithmetic (math_synth, grounded in succ/pred — no host
+ - * /), and checks it equals the RIGHT side. A database can't fake this: the
answer is a computed value, and every op runs on a procedure the brain LEARNED.

    /opt/homebrew/bin/python3.13 exam_math.py
```

### fact_extractor.py
```text
fact_extractor.py — learn by READING (the inverse of production).

The conversation engine turns facts into sentences; this turns sentences back
into facts. For controlled, well-formed text it parses (subject, relation,
object) triples with grammar patterns — no LLM — and resolves "it"/"they" to the
running subject across sentences (the same coreference the conversation loop
uses). Feed it a paragraph; it learns the facts; then the brain reasons over
them.

    fe = FactExtractor()
    fe.extract("An apple is a fruit. It is red. It grows on a tree.")
      -> [("apple","isa","fruit"), ("apple","is","red"), ("apple","grows_on","tree")]
    fe.teach_into(text, conversation_engine)   # read -> stored -> queryable

Pluggable: FactExtractor.extract() is the interface. An LLM-based extractor for
MESSY/open text implements the same method and drops into the same slot —
offline, heavier, noisier — without changing anything downstream. This is the
clean, lightweight, verifiable side; the LLM is the upgrade for open text.

Honest scope: controlled declarative sentences (X is a Y / X has Y / X verbs Y,
with simple coreference). Messy open prose is the LLM's job.
```

### factorizer.py
```text
factorizer.py — the make/break loop's other BREAK half: decompose SOLVED artifacts
into reusable parts, growing the DSL from what the brain has already built.

Composition with a FIXED primitive set has a hand-coded ceiling. Composition + factoring
grows the set: take every formula/program the brain solved, find the repeated structural
shape, promote it to a NEW named primitive, rewrite the library to call it. Next search
reuses it -> solves harder things -> factor again. (This is DreamCoder's wake/sleep
abstraction step, in miniature.)

Expressions are nested tuples: ('*', ('+', 'a', 'b'), 'c'); leaves are var names / numbers.
A discovered primitive Pk(x,y,...) is the most frequent SHARED SHAPE (same operator
skeleton, leaves abstracted to holes). Rewrites are verified to evaluate IDENTICALLY on
random bindings — factoring may never change meaning.

Honest limit: this abstracts over leaf VALUES with a matching operator skeleton (concrete
structural reuse). Deeper anti-unification (abstracting differing sub-shapes too) is the
next rung; this is the first, verifiable step.
```

### feature_learner.py
```text
feature_learner.py — the proposer DISCOVERS which task signals matter (no hand-picked features).

online_proposer2 keyed on a hand-picked signature (out_exceeds_max, neg_in, ...). This removes
that last hand-coded piece: generate a BROAD pool of cheap candidate signals, then LEARN which
ones predict the winning space — from outcomes alone. The predictive features earn weight; the
noise features are ignored. The proposer routes by the signals it discovered itself.

  gen_features : ~16 cheap auto-signals over a task's examples (most are noise)
  per space    : keep a running feature PROFILE of the tasks that space won
  discriminate : a feature's weight = how much it VARIES across space profiles (separates them)
  route        : order spaces by weighted similarity of the task to each space's profile;
                 reward the winner -> its profile sharpens, the useful features surface.

Measured vs the static order: fewer attempts on a mixed workload, AND it reports which
features it FOUND predictive (the discriminative ones emerge with no human naming them).

Honest limit: features are drawn from a fixed generator pool; inventing brand-new signal
TYPES is the deeper rung. But which signals MATTER is now learned, not told.
```

### ground_blend.py
```text
ground_blend.py — concept blending wired into the REAL grounding system.

concept_blend.py proved the idea on toy vectors. This runs it on the actual perceptual
pipeline: the C++ SOM self-organizes raw data, grounding.py grounds concepts as mean SOM
activation maps, and blend() fuses two of those REAL centroids into a new one — then
verifies the blend occupies perceptual space NO existing concept owns (its nearest known
concept is below the same-concept similarity floor). A verified-novel blend is registered,
so recognize() gains a new category grounded in the real SOM.

This is the "mint a new concept region" step on real activations (no C++ change — the
fixed-neuron SOM is untouched; the new concept lives in the Python centroid registry over
its activation space).

Honest limit: blend PROPOSES a novel region and verifies it's unclaimed; whether the
region is USEFUL still needs grounded examples to confirm. Sparse near-binary activation
maps (a known SOM limit) make the blend a union of two regions.

    venv2/bin/python3 ground_blend.py
```

### ground_numeric.py
```text
ground_numeric.py — ground CONTINUOUS quantities, feed them to the policy engine.

grounding.py grounded categories (symbols). This grounds NUMBERS: a perception
encodes measurable quantities (mass, accel); a learned decoder reads them back;
the recovered values are asserted as crisp facts; the C++ PolicyEngine computes a
derived quantity (force = mass*accel) from quantities it PERCEIVED, not was told.

  observation vector -> decode (mass, accel) -> brain.teach_fact -> policy_solve(force)

The decoder is calibrated from a few labeled observations (a grounded "sensor"),
then verified on fresh ones. So the numeric reasoning chain is fed by perception
end to end, and checked against truth.

    venv2/bin/python3 ground_numeric.py
```

### ground_reason.py
```text
ground_reason.py — close the loop: grounded PERCEPTION -> asserted FACT -> REASONING.

grounding.py recognizes a raw observation as a concept. This feeds that straight
into the reasoner: recognizing an object as "metal" asserts (object, isa, metal),
and the ReasoningEngine INHERITS the concept's properties to the object via a
composition rule (X isa Y AND Y property Z => X property Z). So a property the
brain was never told about the object is derived from what it PERCEIVED + what it
knows about the category.

  see raw vector -> recognize 'metal' -> learn (wire7, isa, metal)
                 -> reason: wire7 isa metal, metal property conductive
                 => wire7 property conductive        (grounded, then inferred)

Perception becomes a native reasoning input — the brain knows what it's looking at
AND what follows from that.

    venv2/bin/python3 ground_reason.py
```

### ground_to_binding.py
```text
ground_to_binding.py — grounded perceptions auto-flow into the brain's binding memory.

Closes perception -> memory -> reasoning entirely inside the C++ brain. A perceived
quantity is decoded, then WRITTEN as a vector triple into the brain's own
BindingMemory (the hippocampal store) — not a Python dict. Reasoning then pulls the
fact back out of that memory through a confidence gate and the PolicyEngine
computes with it. The brain stores what it perceives, and reasons from what it
stored.

  perceive -> decode value -> brain.binding.bind(entity, rel, value)   [C++ memory]
  reason   -> brain.binding.query(entity, rel) -> gate -> PolicyEngine.solve

    venv2/bin/python3 ground_to_binding.py
```

### grounding.py
```text
grounding.py — the brain forms its OWN concepts from raw observation, then grounds
symbols on them (a step toward generality: meaning anchored in data, not just told).

Until now facts/symbols were TOLD ("rocket mass 1000"). Grounding is the brain
seeing raw observation vectors, self-organizing them with the SOM into concept
regions, and attaching a symbol to each region — so the symbol "alpha" MEANS "this
part of perceptual space," recognizable from new raw input it was never told about.

  1. observe unlabeled vectors  -> SOM self-organizes (unsupervised structure)
  2. a FEW labeled examples     -> ground a symbol onto each SOM region (sparse, like
                                   a human pointing: "this is X")
  3. new raw observation        -> SOM -> region -> recognized symbol  (grounded!)
  VERIFY: recognition accuracy on held-out observations (does the grounding
  generalize? — the same verifier discipline, now for perception).

Then it connects to the reasoner: perceive raw data -> recognize the concept ->
recall a property of it. The brain knows WHAT it's looking at, then reasons.

    venv2/bin/python3 grounding.py
```

### harden_regress.py
```text
harden_regress.py — correctness regression: lock in the verified OUTPUTS of the core modules.

harden_test checks nothing crashes. This checks the answers are still RIGHT — the production
behaviour we verified, asserted, so a refactor or a C++ port can be checked against known-good
results. Run before every port; a failure means behaviour drifted.
```

### harden_test.py
```text
harden_test.py — surface fragility in the core modules before porting/integration (step 1).

Runs each proven module against EDGE CASES (empty, single, malformed, extreme inputs). A
robust module either handles them or fails cleanly; a crash is a hardening bug to fix. This
is the find-the-fragility pass — fixes follow.
```

### inductive_engine.py
```text
inductive_engine.py — learn rules from data, keep only the verified ones.

Ideas #1 (hypothesis generation) + #3 (simulation/testing) as one honest loop.
Instead of being TOLD every rule, the brain scans observed episodes for patterns
("B tends to follow A"), proposes provisional rules, then VERIFIES each against a
held-out split — promoting the ones that hold and rejecting coincidences.

    il = InductiveLearner()
    promoted, rejected = il.mine(train_episodes, test_episodes)
    il.promote_into(reasoning_engine, promoted)   # discovered rules become usable

The point is the gate, not the guess: generation is cheap (you can always propose
A -> B); the value is rejecting spurious correlations. A pattern that scores 100%
on training but fails on the hold-out set is superstition, not a rule. Honest
scope: co-occurrence rules over symbolic episodes, with support/confidence
thresholds and a hold-out check — it does not infer causation, only association
that REPLICATES. Real causal discovery needs intervention, not just observation.
```

### integral_engine.py
```text
integral_engine.py — symbolic integration: rules SELECTED by form, may fail.

The contrast with differentiation is the point. Differentiation always succeeds:
one rule per node, composed deterministically. Integration must CHOOSE a rule by
the integrand's shape, and can fail outright — many functions (e.g. sin(x^2))
have no elementary antiderivative. That possibility of failure is what makes
integration the SEARCH case rather than the mechanical one.

Each result is verified by DIFFERENTIATING it back with the calculus engine —
integration checked by its own inverse:

    ie = IntegralEngine()
    ie.integrate(("^", "x", 2))      # -> x^3/3   (then d/dx(x^3/3) == x^2)

Honest scope: a bounded ruleset (constants, powers, sums, constant multiples,
basic trig/exp, 1/x). The hard cases — u-substitution, integration by parts —
are where real backtracking search lives (exponential); not built here. Returns
None when no rule in the set applies (honest "not elementary / unsupported").
```

### invariant_miner.py
```text
invariant_miner.py — the brain makes its OWN verifiers (self-growing the envelope).

Verifiers are the envelope: what the brain can check is what it can trust. Until now every
verifier was hand-coded. This mints them from experience, the way a person turns a reliable
regularity into a rule they check against ("if it violates conservation, I'm wrong"):

  1. MINE     candidate invariants that hold across solved examples.
  2. VALIDATE each on HELD-OUT cases — an invariant earns trust only by surviving cases it
              could have broken (kills coincidences that held in the sample).
  3. ADMIT    survivors as cheap checks.
  4. FILTER   a new candidate WITHOUT an oracle: violate an admitted invariant -> reject.
  5. DEMOTE   any invariant that ever rejects a known-correct output (it was spurious).

Arity-general: an example is (args, y) where args is a tuple (scalars auto-wrapped), so the
same machinery mines 1-arg tasks (factorial) and 2-arg tasks (gcd: out divides both inputs,
out <= min). Richer vocabulary = a wider, cheaper pre-filter.

Honest ceiling: invariants are NECESSARY, not sufficient. Violating one => DEFINITELY wrong;
satisfying all != proven correct. A fast error-detector / pre-filter, not a completeness proof.
```

### irregularity_detector.py
```text
irregularity_detector.py — know the boundary: detect domains that have NO checkable regularity.

Self-verifiers reach exactly as far as the open world is REGULAR. The other part — chaotic,
one-off, subjective, genuinely novel — has no invariant to mint, no law to fit. The brain
can't conquer it, but it must KNOW it's there: detect the absence of regularity and abstain
honestly, instead of hallucinating a law that isn't real.

Given (input -> output) data, it checks three things and declares REGULAR vs IRREGULAR:
  1. functional?   same input -> same output (else not even a function -> irregular)
  2. law fits?     a simple law (linear / power) predicts HELD-OUT within tolerance
  3. invariant?    a mined invariant survives held-out

REGULAR  -> the brain can verify here; mine/induce/reason.
IRREGULAR-> no regularity survives -> the brain ABSTAINS (this is the unverifiable part).

This is epistemic humility as a mechanism: the detector is the brain's MAP of its own reach.
```

### knowledge_base.py
```text
knowledge_base.py — scalable knowledge ingestion (the coverage bottleneck).

Coverage = what the brain knows, so growing the brain means INGESTING knowledge.
This is the pipeline: pull triples from multiple curated sources (ConceptNet, or
plain text via the fact extractor), normalize, DEDUPE, persist, and report what's
in there — then load it all into the reasoning brain so coverage on real questions
goes up.

    kb = KnowledgeBase()
    kb.ingest_conceptnet()                 # curated common-sense triples
    kb.ingest_text("A whale is a mammal. It lives in the ocean.")
    kb.into(brain)                         # the brain now knows it
    kb.stats()                             # facts / entities / relations / by-source

Honest scope: ingestion of CURATED, trusted sources — not crawling the web. The
value is a clean, deduped, persistent store and a measurable coverage number, so
"feed the brain more" becomes a concrete, trackable step toward a product.
```

### knowledge_distill.py
```text
knowledge_distill.py — the cloud AI PARSES data into both halves of the brain.

The teacher (qwen-coder) is used not just to make corpus text, but to PARSE a domain into
STRUCTURED knowledge the symbolic brain can ingest and VERIFY — plus sentences the student LM
learns. One teacher, both halves:

  teacher emits, per topic, a strict format:
    OBJECT: <name>
    FACT:  <name> | <property> | <number>
    LAW:   <derived> = <expression over the properties, using + - * / and numbers>
    SENT:  <one plain factual sentence>

  -> FACTs  -> taught to the symbolic reasoner (ReasoningEngine.learn)
  -> LAWs   -> parsed to a policy, VERIFIED (must compute on the object's facts) then ADMITTED
  -> SENTs  -> corpus -> the student LM learns from the same knowledge

Membrane holds: the teacher PROPOSES facts/laws; the brain VERIFIES a law computes before
admitting it (a law that doesn't evaluate on the facts is rejected). The student learns the
language; the symbolic core learns the verified structure. Both from one parsed source.
```

### knowledge_engine.py
```text
knowledge_engine.py — hardened Knowledge layer (milestone #1, productionized).

The brain_repl demo proved the mechanism: learn facts online, derive the
unstated, explain the chain. This is the hardened version — a clean, validated,
persistent API around the binding memory, with input validation, idempotency,
graceful handling of unknowns and cycles, and save/load.

    kb = KnowledgeEngine()
    kb.learn("alice", "manages", "bob")
    kb.learn("bob", "manages", "carol")
    kb.ask("alice", "manages", hops=2)        -> ("carol", confidence)
    kb.explain("alice", "manages")            -> "alice -> bob -> carol"
    kb.save("org.json"); KnowledgeEngine.load("org.json")

Facts are the source of truth (persisted as JSON); the binding memory is a
derived index, rebuilt deterministically by replay — so save/load is exact and
does not depend on C++ serialization.
```

### knowledge_pack.py
```text
knowledge_pack.py — fuse curated sources into one shippable, measured knowledge file.

Builds the brain's knowledge from the vetted core seed + ConceptNet (+ any TSV /
N-Triples dumps you pour in), dedupes, saves one JSON pack, and reports stats and
coverage on a sample question set. This is the "feed the brain" step turned into a
single artifact you can ship and reload.

    python3 knowledge_pack.py                 # build + report
    kb = build_pack("knowledge_pack.json", extra_tsv=["wikidata_subset.tsv"])

Honest scope: structured, curated sources. The full Wikidata dump (terabytes of
opaque Q/P codes) needs label resolution and is not a laptop job — but a labeled
subset exported as TSV/N-Triples drops straight in.
```

### learn_by_reading.py
```text
learn_by_reading.py — the autonomous self-extension loop (the "learns logic" thesis).

    corpus (text)  ->  LLM extracts numeric examples  ->  guided induction
    discovers the formula  ->  held-out VERIFY  ->  STORE as a policy in the Brain
    ->  the Brain can now answer NEW instances of a law it learned BY READING.

The LLM is only the reading EYE (text -> numbers); the discovery is the brain's own
guided induction; the gate is held-out verification; the result lives in the C++
Brain (brain.policy_add) and is used by brain.policy_solve. Nothing is stored unless
it generalizes — induction proposes, verification disposes.

Run with the interpreter that has brain2 (and a model, or the stub):
    venv2/bin/python3 learn_by_reading.py            # deterministic stub reader
    venv2/bin/python3 learn_by_reading.py --real     # real local qwen3:1.7B reader
```

### learned_guidance.py
```text
learned_guidance.py — hardened Learned search guidance (milestone #5).

The reasoning that improves with experience, made a clean reusable module. A
LearnedHeuristic fits a linear estimate of cost-to-goal over state features,
trained from solved instances, then guides the hardened search engine. The two
guarantees the tests pin: guided search stays CORRECT (still solves), and it
expands far fewer states than blind search.

    h = LearnedHeuristic(features)
    h.train(collect_examples(EightPuzzle, scramble, manhattan))
    solve(EightPuzzle(start, hfn=h))      # same engine, now guided

Domain-agnostic: give it a feature function and solved instances. Demonstrated
on the 8-puzzle (where it rediscovers the Manhattan heuristic from experience
and cuts search ~100x). Persistable; deterministic given a seed.
```

### llm_adapter.py
```text
llm_adapter.py — plug a local LLM into the Eyes and Mouth slots.

The brain is the mind; the LLM is only IO. This implements neuro_bridge's
Eyes/Mouth with a local model (Ollama / llama.cpp on a Mac), with NO training —
the model is used off the shelf as a translator:

  Eyes : messy language -> a structured Query the brain can reason over
  Mouth: a verified Answer -> a fluent sentence (constrained to the answer)

Design keeps the exact path exact: math notation still goes through the
deterministic recursive-descent parser; the LLM is only the FALLBACK for phrasing
the parser can't handle. So adding the model never makes a math answer less
reliable — it only widens what the eyes can read and smooths what the mouth says.

    eyes = LLMEyes(OllamaClient("qwen2.5"))
    mouth = LLMMouth(OllamaClient("qwen2.5"))
    Mind(eyes, Brain(), mouth)

Honest limit: prompt-constraining the mouth reduces but does not PROVE no
hallucination — the verified content is always in the Answer for audit, and hard
guarantees need constrained decoding. So verified/unknown answers fall back to the
deterministic grammar mouth; the LLM only polishes language-domain answers.
```

### llm_extractor.py
```text
llm_extractor.py — learn from a corpus: prose -> triples, grounded, then ingest.

Closes the "train on a corpus" loop honestly. A trusted text is read by the LLM
into (subject, relation, object) triples; each triple is GROUNDED-checked (its
terms must actually appear in the source) before it is kept, so the model can't
smuggle in facts the text never stated. Same `extract(text)` interface as the
grammar fact extractor, so it drops straight into `knowledge_base.ingest_text`.

    kb.ingest_text(textbook_paragraph, extractor=LLMExtractor(OllamaClient("qwen3:1.7B")))

Honest boundary: this trusts the SOURCE and verifies the EXTRACTION (faithful, not
hallucinated). It does NOT verify the source is true — feed it curated, trusted
text. Truth-checking a claim needs corroboration or experiment, not parsing.
```

### loop_synth.py
```text
loop_synth.py — the brain synthesizes ALGORITHMS (loops), not just formulas.

brain_codegen built straight-line formulas. This adds CONTROL FLOW: the brain
searches a small imperative DSL — an accumulator loop — for a program matching
input/output examples, verifies on held-out cases, and renders it to Python. No
LLM: the brain finds the algorithm (init + loop range + update step), the renderer
is mechanical, the code is re-verified.

  DSL:  acc = INIT ; for i in RANGE: acc = UPDATE(acc, i) ; return acc
  search INIT x RANGE x UPDATE  ->  fits examples + held-out  ->  render -> verify

Covers the fold/accumulator family (factorial, sums, sum-of-squares, max, count) —
a broad class of real algorithms with loops. Not arbitrary programs (no recursion
trees / sorting), but genuine control-flow synthesis beyond formulas.

    python3 loop_synth.py
```

### loop_synth2.py
```text
loop_synth2.py — richer algorithm synthesis: two accumulators + conditionals.

loop_synth did single-accumulator folds. This adds two control structures, taking
more algorithm classes away from the LLM and giving them to the verified synthesizer:

  TWO-STATE:   a,b = ia,ib ; for i in RANGE: a,b = NA(a,b,i), NB(a,b,i) ; return a|b
  COND-FOLD:   acc = INIT ; for i in RANGE: if COND(i,n): acc = UPDATE(acc,i) ; return acc

Search each space for a program fitting input/output examples, verify on held-out,
render Python. No LLM. Covers Fibonacci (two-state), count-of-divisors / sum-of-evens
(cond-fold) — branching / multi-state algorithms, not just folds.

    python3 loop_synth2.py
```

### loop_synth3.py
```text
loop_synth3.py — while-loops + early-return synthesis (GCD, primality, factors).

The next control structures the brain reclaims from the LLM:

  WHILE (two-arg):  while COND(a,b): a,b = NA(a,b), NB(a,b) ; return a|b   (Euclid GCD)
  EARLY-RETURN:     [if PRE: return p] ; for i in RANGE: if COND(i,n): return R1
                    ; return R2                                            (primality, factor)

Search each space for a program fitting input/output examples, verify on held-out,
render Python. No LLM. while-loops are condition-terminated (not range-bounded) and
early-return exits mid-loop — genuinely new shapes beyond folds.

    python3 loop_synth3.py
```

### loop_synth4.py
```text
loop_synth4.py — list/string inputs + nested loops (sum_list, max, contains, sort).

New input type (sequences) and a new control structure (nested loops). To keep the
search from exploding, each algorithm class is a PARAMETERIZED TEMPLATE with a tiny
parameter set — the brain searches the parameters, not an open program space:

  FOLD     : acc=INIT; for x in lst: acc=UPDATE(acc,x); return acc   (sum,product,max,min)
  MEMBER   : for x in lst: if x==t: return True; return False        (contains)
  NESTED   : for i: for j>i: if lst[i]==lst[j]: return True; ...      (has_duplicate)
  SORT     : bubble sort, comparator searched                        (ascending/descending)

Search the parameters to fit examples, verify held-out, render Python. No LLM.

    python3 loop_synth4.py
```

### math_chat.py
```text
math_chat.py — ask the math engines in plain notation.

Wires the exact math parser to the calculus / integral / algebra engines and
routes by intent word, so you can type the request instead of building tuples:

    > differentiate sin(x^2)      d/dx = cos(x^2)*(2*x)   [sin, power, chain]
    > integrate cos(x)            int = sin(x) + C        [checked]
    > solve 2*x + 3 = 7 for x     x = 2                   [(7-3)/2, verified]

Honest scope: math notation only (a formal grammar — parsing is exact here),
intents differentiate / integrate / solve. Not natural-language word problems.
```

### math_parser.py
```text
math_parser.py — parse math notation into the engines' expression tuples.

Math notation is a formal grammar, so this is an EXACT recursive-descent parser
(not the controlled-approximation parsing that natural language needs). It turns
"sin(x^2)" / "2*x + 3 = 7" into the nested tuples that calculus / algebra /
physics / integral engines consume.

    parse("sin(x^2)")     -> ("sin", ("^", "x", 2))
    parse("2*x + 3 = 7")  -> ("=", ("+", ("*", 2, "x"), 3), 7)

Grammar (precedence low -> high):
    equation := expr ('=' expr)?
    expr     := term (('+' | '-') term)*
    term     := power (('*' | '/') power)*
    power    := unary ('^' power)?            # right-assoc
    unary    := '-' unary | atom
    atom     := number | func '(' expr ')' | ident | '(' expr ')'

Honest scope: explicit operators (2*x, not 2x), the functions sin/cos/exp/ln.
```

### math_synth.py
```text
math_synth.py — the brain LEARNS arithmetic instead of calling it.

The C++ core has frozen skills: MATH_MUL is host `a * b`, MATH_QUAD a baked
quadratic solver (logic_engine.hpp). The basal ganglia only learns WHICH frozen
op to route to — never what multiply IS. Math is called, not known.

This grounds arithmetic from the floor, reusing the PROVEN synthesis engine
(tree_reason.solve — the same A* program_synth uses). The substrate the string
DSL lacked:

  floor atoms:  Z (zero), S (successor +1), P (predecessor -1, truncated)
  variables:    a, b, r   (operands + the recursive result)
  combinator:   primitive recursion, on EITHER argument (the `repeat` we added):
     recurse on a:  f(0, b) = BASE(b)   ;  f(Sa, b) = STEP(a, b, f(a,b))
     recurse on b:  f(a, 0) = BASE(a)   ;  f(a, Sb) = STEP(a, b, f(a,b))

The synthesiser searches BASE+STEP whose function matches the I/O examples —
VERIFIED, then checked on held-out large inputs (generalisation, not memory).
Solved functions are CACHED into the basis (library abstraction); each new op is
built from earlier learned ones. That cache is the reattach unit.

Claim under test: from only {S, P} + recursion (no host + - * /), synthesise
add, sub, mul, pow — each grounded in the last. If yes, the brain LEARNED math.

    /opt/homebrew/bin/python3.13 math_synth.py       # proof + library
    from math_synth import LearnedArithmetic         # importable module
```

### means_ends.py
```text
means_ends.py — Phase 2 + 3: the problem-solving EXECUTIVE with a LEARNED,
STORED policy memory and a conjecture -> verify -> admit loop.

Phase 2 (the spine): Newell & Simon's means-ends analysis over a BLACKBOARD.
  Take a goal. If a source answers directly, done. Otherwise a source that knows
  HOW says what it's MISSING, posts those as sub-goals, recurse, compose. Every
  solved sub-goal is MEMOIZED (tabling). Sources COMMUNICATE through the shared
  blackboard — they read/write it, they do not fuse (the crisp/fuzzy membrane).

Phase 3 (this version):
  * Policies live in a PolicyMemory — a stored, persistable, learnable table,
    SEPARATE from the fact KB. Each policy is a serializable tuple-formula over
    named inputs (evaluated by the real physics_engine.ev). "Rules in memory,
    not hardcoded boards."
  * The PolicyLearner CONJECTURES new policies by composing existing ones
    (inlining a sub-policy), then VERIFIES the conjecture numerically against the
    executive's own step-by-step derivation before ADMITTING it. A wrong
    conjecture is REJECTED at the gate. Induction proposes; verification disposes.

  force = mass*accel ; power = force*speed   --conjecture-->  power = (mass*accel)*speed
  verified against the 2-step derivation -> admitted -> future power queries are
  one shot (the composition is "compiled" into a new stored policy).
```

### mouth.py
```text
mouth.py — the OWNED generative mouth: structure -> English, learned like a child.

Not an LLM. The insight: a learned Template maps tokens <-> slots, so the SAME grammar the
brain acquired by comprehension runs BACKWARD to produce. The mouth speaks with the
constructions it learned to parse, and gets more fluent as it reads more — exactly child
language acquisition (comprehension grammar = production grammar).

Membrane: the mouth only renders VERIFIED structure (an Event the membrane admitted, a fact
the store owns). It never invents content — grammar dresses truth. Fluency is child-grade
early (imperfect morphology, like a kid saying 'goed'); it improves with learned templates and
morphology, and a tuned LLM stays an anytime fluency fallback.

Two surfaces:
  * say_event(Event)         — agent verb(s)/did-not-verb patient, tense + polarity realized
  * say_fact(e, rel, v, tm)  — bidirectional: fill a learned statement template's slots
```

### nested_parser.py
```text
nested_parser.py — compositional comprehension: a query INSIDE a query, and conditionals.

structural_parser handled flat shapes (single/compound/compare). This adds the next rung:
NESTED and CONDITIONAL structure, where one query's answer feeds another.

  superlative entity : "the force of the HEAVIEST object"  -> resolve heaviest = argmax(mass)
                        over entities, THEN force of that entity. The entity slot is itself a
                        sub-query.
  conditional        : "if the rocket's mass is greater than 500 then what is its speed" ->
                        evaluate the condition; only if it holds, answer the consequent (with
                        pronoun 'its' bound to the condition's entity).

Every sub-result is computed by the crisp solver, so the whole composite is verified end to
end; a part that doesn't verify makes the whole abstain. Propose-and-verify, recursively.

Honest limit: two compositional forms (superlative, single-condition if/then) — not a full
recursive grammar. Deeper nesting and boolean conditions (and/or) are the next rung.
```

### neural_lm.py
```text
neural_lm.py — grow the probabilistic pillar: a NEURAL sequence model over the brain's corpus.

prob_compute (n-gram) only matches sequences it has SEEN; an unseen context backs off blindly.
A neural LM with learned EMBEDDINGS generalizes: words used the same way (speed/velocity) get
similar vectors, so the model predicts sensibly on contexts it never saw and understands that
two words play the same role. That generalization IS the comprehension leap — same paradigm,
better engine, still OWNED and internal (pure numpy, trained on the brain's own text).

  * a feedforward neural LM (embed previous k words -> hidden -> softmax over vocab)
  * trained by SGD on the corpus the brain read
  * GENERATES sentences, gives UNCERTAINTY, and its embeddings CLUSTER synonyms it was never
    told are related — learned purely from how they're used.

Membrane unchanged: probabilistic PROPOSES, symbolic VERIFIES. Honest ceiling: fluency scales
with corpus + model size; this proves the neural paradigm self-contained, not frontier scale.

    venv2/bin/python3 neural_lm.py
```

### neural_lm_torch.py
```text
neural_lm_torch.py — the REAL owned LM for Mac training: a small decoder-only Transformer on
Apple-Silicon GPU (MPS). Same interface as neural_lm (train/generate/dist) so it drops into
train_pipeline as the student; scales far past the numpy proof.

Word-level, causal Transformer. Config (dim/layers/heads/ctx) tiny by default, Mac-trainable;
raise for a real run. device = mps (Apple GPU) if available, else cpu. Given the symbolic
core offloads facts + reasoning, a SMALL LM here (10-100M params) is the practical target and
is Mac-feasible.

Requires PyTorch:  pip install torch    (MPS ships with the standard macOS wheel)

    venv2/bin/python3 neural_lm_torch.py
```

### neuro_bridge.py
```text
neuro_bridge.py — the IO contract: LLM is the eyes and mouth, the brain is brain.

The brain (controller) operates ONLY on structured Query/Answer — it never sees
raw text. Eyes turn language into a Query; Mouth turns a verified Answer into
language. The LLM is just one implementation of Eyes/Mouth and drops into those
named slots without touching cognition; the v0 here uses the exact symbolic
parsers we already built.

    mind = Mind(RuleEyes(), Brain(), GrammarMouth())
    mind.respond("differentiate sin(x^2)")   # eyes -> brain -> mouth

Flow:  text --Eyes--> Query --Brain--> Answer --Mouth--> text
The brain decides content; the LLM only translates in and out, so it cannot
invent facts. Coverage = what the brain knows; everything it answers is verified
or honestly flagged unknown.
```

### nl_front.py
```text
nl_front.py — the runtime language front: lexical matcher + DISTILLED student.

Closes the distillation loop. A question's relation is resolved by:
  1. the lexical matcher (nl_query) when it has a STRONG hit (exact name / prefix /
     close synonym) — high precision, zero training;
  2. otherwise the distilled student (student_trainer, trained on the cloud
     teacher's varied phrasings) — generalizes to wording the matcher misses.
Entity is lexical. The resolved Need goes to the verified executive (facts +
policies + proposer). The big teacher never runs here — only its distilled student.

So the system speaks both the phrasings you hand-listed AND the ones the teacher
taught the student, with a verified number out the end and no LLM at inference.
```

### online_proposer.py
```text
online_proposer.py — fix the weakest part: a proposer that LEARNS from every verified outcome.

The synth engine's proposer is STATIC: ROUTES tries the synthesis spaces in a fixed order
(e.g. list-tasks always try 'list' then 'dp'), so a workload whose answers live in the
second space wastes an attempt on every single task. The proposer never learns.

This wraps the SAME backends + the SAME verifier, but orders the spaces by LEARNED priors
and rewards the space that actually solved each task. Over a session the proposer adapts to
the workload: spaces that keep working get tried first, so attempts-per-solve drop. The
fix for the propose/verify asymmetry — generation that improves from experience.

Measured against the static order on the same tasks: fewer total backend attempts, with no
loss of solutions (same verifier gates every candidate). Honest limit: learns space-ORDER
from outcomes; richer per-task features (beyond kind) are the next rung.
```

### online_proposer2.py
```text
online_proposer2.py — richer features (transfer) + loop signals (sandbox/refuter) for the proposer.

online_proposer learned space-order keyed only by `kind`. That can't separate task FAMILIES
inside one kind: e.g. 'list' holds both sort tasks (solved by the 'list' space) and
subarray tasks (solved by 'dp'). Keyed by kind alone, learning one helps and HURTS the
other. This keys priors on a richer feature SIGNATURE of the task (list-vs-scalar output,
negatives, output-grows, output-exceeds-max-element), so each family learns its own best
space — and a NEW task transfers to the right space by matching signature, not by repetition.

Loop signals feed the same priors:
  sandbox/stress survived -> reward the space (this shape holds)
  refuter: candidate broke under stress -> penalize the space (this shape breaks here)

Measured vs kind-only and vs the static order on a MIXED workload (sort + subarray
interleaved): the feature proposer solves both families near-optimally; kind-only can't.

Honest limit: a hand-picked feature set; learning the features themselves is the next rung.
```

### parse_template.py
```text
parse_template.py — symbolic sentence templates: the crisp grammar unit.

A Template is to language what a Policy is to physics: a stored, serializable, verifiable
rule. ("w", word) items must match exactly; ("slot", name, type) items bind a typed value;
("any",) skips one token. Matching is exact-length — no fuzzy scoring here (the membrane:
fuzz lives in the grounder). Induction (slot_example + anti_unify) mirrors factorizer's move
for formulas, applied to word sequences. (Plan Phase A, Tasks 1/2/4.)
```

### physics_engine.py
```text
physics_engine.py — apply taught physics laws, solving for ANY variable.

A law is one equation (lhs_symbol = rhs_expression). Given values for all but one
variable, the engine isolates the unknown symbolically — inverting operations
outward (x -> /, + -> -, ^n -> ^(1/n)) — then evaluates, showing the rearranged
formula and the numbers. It applies laws; it does not invent physics.

    pe = PhysicsEngine()
    pe.add_law("newton2", "F", ("*", "m", "a"))      # F = m*a
    pe.solve("newton2", "a", F=12, m=3)              # a = F/m = 12/3 = 4

Reuses the expression-tuple representation (and renderer) of calculus_engine. The
isolation routine is the kernel of the algebra solver too — generalized there.
```

### planning_engine.py
```text
planning_engine.py — hardened "Knowledge + search joined" (milestone #4).

The first layer that composes two already-hardened layers: actions and their
preconditions/effects are stored as facts in the KnowledgeEngine, and the
hardened search engine plans a valid ordering to reach a goal — then explains
it. Genuine multi-precondition planning (an action can require several inputs
from different branches), not single-relation chaining.

    pe = PlanningEngine()
    pe.define_action("smelt", requires=["ore"],          produces=["iron"])
    pe.define_action("chop",  requires=["axe"],          produces=["wood"])
    pe.define_action("forge", requires=["iron", "wood"], produces=["sword"])
    plan = pe.plan(have=["ore", "axe"], goal="sword")
    plan.found     -> True
    plan.explain() -> chop -> smelt -> forge, with data-flow

Actions live in the KnowledgeEngine (persisted, validated); the plan comes from
the optimal, deterministic search engine. Both hardened, with their own tests.
```

### policy_induction.py
```text
policy_induction.py — the brain DISCOVERS a policy from examples (induction).

Until now the executive only COMPOSED hand-given policies (conjecture->verify).
This is the next rung: given examples (input columns + a target column), INDUCE the
formula by bounded symbolic regression over {+,-,*,/}, then VERIFY it on held-out
examples before admitting it as a Policy. Induction is unsound (a fit on N rows is
a guess), so the held-out check is the gate — induction proposes, verification
disposes, exactly the discipline everywhere else.

    induce(rows, ["mass","accel"], "force")  ->  ("*","mass","accel")  (verified)

Honest scope: formulas up to a small tree depth over the given inputs (+ constants
0.5, 2). Deeper formulas (e.g. 1/2 m v^2) blow up blind enumeration — that's where
a proposer-GUIDED induction goes next (the same premise-selection idea), not built
here.
```

### policy_proposer.py
```text
policy_proposer.py — the Proposer wired onto the executive's policy memory.

means_ends.py gave each target ONE policy, so there was nothing to choose. Real
reasoning has MANY ways to reach a goal, most of them dead ends for the facts you
actually have. The Proposer is what makes that tractable: instead of trying
policies blindly (and wasting a whole sub-derivation on one whose leaf inputs
aren't available), it scores each candidate by how GROUNDABLE its inputs are and
tries the likely winner first. This is premise selection — the same idea proven
at 5.4x in program_synth_tree, now over the policy store.

The score here is a structural feature (fraction of a policy's inputs that bottom
out in known facts). A trained proposer would weight several such features; this
one feature is already the signal, and the membrane is intact — the proposer only
ORDERS the search, the verifier-backed executive still produces the answer.

  power = force*speed   OR   power = energy/time     (two policies, one target)
  facts: mass, accel, speed   (time is MISSING)
  blind tries energy/time first -> wastes the whole energy subtree -> backtracks.
  proposer scores force*speed higher (all inputs groundable) -> no waste.
```

### prob_compute.py
```text
prob_compute.py — the missing THIRD pillar: probabilistic computing over the brain's own data.

The brain had symbolic (exact, verified) and fuzzy (vectors, similarity) computing. It lacked
the PROBABILISTIC mode — distributions, sequences, sampling, uncertainty, GENERATION. An LLM
is exactly this, at scale. But the brain already HAS the data (co-occurrence over a corpus it
read); it only lacked the engine. This is that engine, built from the brain's own text:

  * FORM SENTENCES  — sample next word from P(word | context), an n-gram model with backoff.
                      Language generated from the brain's own data, no external LLM.
  * UNCERTAINTY     — every prediction carries a distribution + entropy (confidence). This is
                      the graded "how sure am I" the crisp store never had.
  * MEMBRANE        — probabilistic PROPOSES a sentence; the symbolic core VERIFIES any
                      checkable claim in it. Fuzzy/probabilistic proposes, crisp disposes.

This is the RIGHT type of computing for open language, OWNED and internal. Honest ceiling:
quality scales with corpus + model order; an n-gram forms plausible local sentences, a neural
sequence model forms better ones, frontier fluency needs frontier scale. The paradigm is what
matters — the brain can now generate, not just match.
```

### program_synth.py
```text
program_synth.py — write code from examples, by search, verifiably.

Give it input/output examples. It SEARCHES a small DSL of string operations
for a program that reproduces every example, then applies that program to new
inputs to show it learned the transformation — not memorized the answers.

This is the code version of tree_reason: state = a partial program, operators =
DSL primitives, goal = "the program reproduces all examples." The search finds
the shortest program that passes. The result is verifiable by construction — it
was searched to satisfy the spec, not guessed (which is what makes an LLM's code
plausible-but-wrong). Same engine as algebra and the bridge puzzle; only the
operators and goal differ.

Honest scope: it writes programs in the DSL given here (string ops). Richer
programs need a richer DSL (more operators) and, past a few steps, learned
search guidance — exactly the tree_learn pattern. It does not invent new
primitives. Within the DSL, every program it returns is correct on the spec.
```

### program_synth_guided.py
```text
program_synth_guided.py — synthesis that LEARNS which operators to try.

program_synth.py searches the program space blindly: with a richer DSL and
longer programs it explodes (k operators, depth d => k^d programs). This adds
the learned guidance — the honest home of "a statistical model guides the
brain's choices."

It works the way tree_learn did for the 8-puzzle, now over PROGRAMS:
  1. generate many solved synthesis tasks (random program + inputs -> examples),
  2. learn, from the example features, a PRIOR over which operators a spec like
     this tends to need (a linear model, fit by least squares),
  3. guide the search by that prior: try likely operators first
     (cost of an operator = -log prior, so best-first = most-probable program).

Result: the guided search explores far fewer programs than blind search and
solves harder tasks within the same budget — and the prior was learned from its
own solved experience, not hand-coded. Same idea as a policy network guiding
program search (DreamCoder), here lightweight and on CPU.
```

### program_synth_policy.py
```text
program_synth_policy.py — a goal-conditioned policy guides each next op.

The marginal prior (program_synth_guided.py) only knew which operators a spec
tends to need overall — a weak ~1.8x. This is the full idea: at every step the
policy looks at WHERE THE COMPUTATION IS NOW versus the target, and scores which
operator best closes the remaining gap.

Concretely, for a partial program it runs that program on the inputs to get the
intermediate values, computes features of (intermediate -> target), and the
learned policy scores each next operator from THAT. It is trained on prefixes of
solved programs: for every step of a known solution, "given this intermediate
state, the next op was X." So it learns to drive the computation toward the
target one step at a time — a goal-conditioned policy over the synthesis state,
exactly "a statistical model guides each choice as you build."

It also prunes provably-dead branches (a partial program that already errors
can never be completed). Compared head-to-head with blind search and the
marginal prior on the same deep tasks.
```

### program_synth_tree.py
```text
program_synth_tree.py — the policy as a decision TREE.

The reasoning engine is a search tree; its guide should be a tree too. The
linear policy capped at ~3.8x because it can't model feature INTERACTIONS —
"if the intermediate still has spaces AND the target is short -> initials" is
not a weighted sum. A decision tree splits on exactly those interactions.

It is also the right model for this project: lightweight, fast (traverse, no
forward pass), trains with no backprop (greedy splits), dependency-free, and
INTERPRETABLE — the policy is readable if-then rules.

Same goal-conditioned setup as program_synth_policy: at each step, features of
(intermediate -> target), the tree predicts a distribution over the next op,
and best-first search follows it (dead branches pruned). Compared head-to-head
with blind search and the linear policy on the same deep tasks and features.
```

### query_planner.py
```text
query_planner.py — let the brain DECOMPOSE a question, then reason.

Instead of scattered "if 'how' in text" intents, this turns a (possibly
multi-part) question into a STACK of small structured queries, answers each over
the reasoning engine, and composes one reply. The pipeline mirrors how you'd
break the question down by hand:

  intent      : question / statement / ... (from the AppraisalEngine)
  quantifier  : how many -> count | which -> which | ways -> list | else single
  subject     : the entity in focus (vitamin_c)
  relation    : the predicate asked about (provides / helps), matched or inferred
  -> push each sub-question on a stack, solve, then COMBINE the answers.

Honest scope: the slot extraction is still controlled grammar (the comprehension
wall is unmoved) — but the structure, decomposition, and composition are real and
replace the ad-hoc intent routing with one mechanism.

    qp = QueryPlanner(engine)
    qp.answer("in how many ways can we get vitamin C? "
              "which fruit provides vitamin C?")
```

### read_book.py
```text
read_book.py — hand the brain a BOOK; it reads it ITSELF.

This is the composition the scattered reading modules were missing: one entry that
streams raw prose through the grammar, and calls the LLM ONLY for fragments the grammar
cannot yet parse — needing it LESS as it reads (measured decay). The wiring:

    raw text ─► sentence stream ─► ReadingLoop.parse
                                      │  parses (grammar knows it)  ─► VERIFIED fact  (NO LLM)
                                      │  miss ─► teacher(sentence) [LLMExtractor] ─► triple
                                      │            └─ induce a TEMPLATE ─► next time: NO LLM
                                      ▼
                              escalation curve  (falls = LLM-need shrinking)

The membrane still disposes: a taught fact enters the crisp store only if it admits; the
LLM only ever PROPOSES (and its triple is grounded to terms that appear in the text). The
honest number is the decay curve — how much the brain still needs the teacher, chapter by
chapter. It never reaches zero for genuinely novel structure (the autoformalization wall),
but it should fall, and that fall is the whole claim.

    venv2/bin/python3 read_book.py                       # offline proof (stub teacher)
    venv2/bin/python3 read_book.py data/raw_ssc9.txt qwen3:1.7B   # real: Ollama teacher
```

### reading_loop.py
```text
reading_loop.py — grow language from raw text, not from teacher-made pairs.

The pipeline: template-parse a sentence -> build an Event -> event_verify disposes -> on
ADMIT the parse re-enters template induction. Three disciplines make this safe to run
unattended:

  * Anti-collapse gate — ONLY verified parses (ADMIT) and trusted teacher labels feed
    induction. A rejected/abstained parse never trains the grammar, so the loop can't
    hallucinate itself into a corner.
  * Fragment-level active learning — a sentence the grammar can't parse escalates to the
    teacher; teacher labels buffer per-relation and induce a new template once >=2 exist
    (so anti-unify + a real held-out check run). Teacher touched per-miss, not per-sentence.
  * Measurable decay — escalation rate is tracked; as templates accumulate it must fall.
    That number is the honest 'is the teacher still needed' signal.

(Open-language track, Gap 2 — uses Gap 1's Event + membrane and Gap 3's context stack.)
```

### reasoning_engine.py
```text
reasoning_engine.py — hardened Reasoning layer (milestone #2, productionized).

The Knowledge layer retrieves facts and chains a SINGLE relation (transitive
closure). Reasoning adds what it can't: COMPOSITION RULES across DIFFERENT
relations — "X parent Y and Y parent Z => X grandparent Z" — derived by
backward chaining and explained with the rule that fired.

    re = ReasoningEngine()
    re.learn("tom", "parent", "sam"); re.learn("sam", "parent", "kid")
    re.add_rule("parent", "parent", "grandparent")
    re.ask("tom", "grandparent")    -> ("kid", "tom parent sam AND sam parent kid => ...")

Builds on the hardened KnowledgeEngine (facts grounded in the binding memory);
the rule layer composes relations on top. Same hardening discipline: input
validation, cycle safety, persistence, explanations.
```

### reasoning_suite.py
```text
reasoning_suite.py — the brain's real report card.

Perplexity asks "does the output match the corpus bit-for-bit?". That is the
wrong test for a system meant to *understand*, not memorize. This suite asks
the right question: given strictly less than the answer, can it DERIVE the
rest, on cases it was never shown? That is the difference between inference
and pattern-matching, and it is exactly where a logic engine can beat an LLM.

Every test stores only PRIMITIVE facts and scores only DERIVED conclusions
(never directly stored). For each correct derivation it also prints the chain
in words — the brain showing its actual work, because the reasoning is an
explicit traversal, not hidden weights.

Tests:
  1. transitive_inference  — store adjacent A>B, B>C; derive A>C (k hops),
                             with DISTRACTOR facts present so retrieval must
                             discriminate, not just walk a clean graph
  2. depth_curve           — accuracy vs hop distance (1=stored .. K=derived)
  3. noise_robustness      — query with perturbed vectors: is it real retrieval
                             or brittle exact-match lookup?
  4. relation_composition  — store parent links; derive ancestor (novel pair)

Pure Python over the built module; no training of the LM required.
```

### refute_synth.py
```text
refute_synth.py — wire the refuter into synthesis: self-correct an overfit with a DIAGNOSIS.

synth_engine.solve picks a program that fits the given examples. If the examples are
skewed (e.g. all-positive), it can return an overfit that passes them but is wrong
elsewhere. stress() flags that with a single counterexample. The refuter does better: it
returns WHERE the rule breaks (scope), and we feed that counterexample back as a new
constraint and re-synthesize — the loop corrects itself, with a reason, not just a retry.

  loop: solve(examples) -> refute(vs oracle) -> if it breaks, add the counterexample to
        examples and re-solve; stop when robust (or out of tries).

This is stress-in-the-loop upgraded from "reject" to "diagnose + repair", and it connects
the new break-engine to the synthesizer the brain already runs.

Honest limit: needs an oracle to refute against (same bound as all verification here); the
repair only works if the correct program is reachable in the DSL once the example is added.
```

### refuter.py
```text
refuter.py — the make/break loop's BREAK half: decompose a built rule by attacking it.

The brain composes (synth/induction) and the verifier confirms ("does it pass?").
The refuter is the verifier turned aggressive: "WHERE does it FAIL, and where does it
still HOLD?" Breaking a rule is how you find the gap a NEW rule must fill — so this is
both a diagnostic (pinpoint an overfit) and the generator of the next conjecture.

Given a program (code defining f) or any callable, plus an oracle and a task kind:
  - hunt a counterexample (random + structured edge probes),
  - characterise the VALID SCOPE (for int1, the integer ranges where it holds; for
    lists, which structural property breaks it — empty / singleton / negatives /
    duplicates / sorted).

Honest limit: like everything here, refutation needs an oracle. No oracle -> can only
report "unbroken in N samples", not "correct". It finds where two things DISAGREE.
```

### semantic_depth.py
```text
semantic_depth.py — extend the concept SET from definitions, not just map to a fixed one.

context_embed maps a paraphrase to an ALREADY-KNOWN concept. Semantic depth is the next
step: when a genuinely NEW concept appears, learn it (from a definition that composes known
concepts), VERIFY it, and add it — so comprehension grows past the fixed vocabulary instead
of abstaining. This is the membrane again: a definition PROPOSES a new concept; the crisp
solver VERIFIES it computes before it is admitted.

  "momentum is mass times speed"  -> new relation momentum = mass * speed -> verify on a
  known entity -> admit. Now "what is the momentum of the rocket" routes and computes,
  and a paraphrase of the new word (via context) routes to it too.

So the brain's vocabulary EXTENDS itself from language, every new concept verified by
computation. Honest limit: definitions must compose KNOWN concepts with a known operator
(times/over/plus/minus) — grounding a concept from raw perceptual data (ground_blend) is the
complementary path; truly primitive new concepts still need that.
```

### semantic_memory.py
```text
semantic_memory.py — the memory doing what a dict cannot.

The binding memory's strength is similarity-based retrieval, but with random
token vectors that strength is dead (everything is exact-match, so a dict wins).
This gives it REAL embeddings (GloVe): now semantically-near tokens are close in
vector space, so it GENERALIZES — answer a query about "car" from a fact stored
about "automobile", because they mean nearly the same thing. A dict returns
nothing; the vector memory returns the answer.

    sm = SemanticMemory()
    sm.learn("automobile", "has", "engine")
    sm.ask("car", "has")          -> ("engine", confidence)   # never stored "car"
    sm.similar("car")             -> ["automobile", "vehicle", ...]

Honest limit: it is only as good as the embeddings. GloVe-50 captures strong
synonymy (car~automobile, dog~puppy) but is noisy on subtler pairs and (a known
quirk) rates antonyms as similar. Better embeddings -> better generalization.
```

### server.py
```text
server.py — Brain2 FastAPI Server
Live cognitive engine with chat, causal chain reasoning, confidence-gated answers,
WebSocket state broadcasting, and continuous daydreaming.
```

### stress_synth.py
```text
stress_synth.py — stress-IN-THE-LOOP synthesis: self-corrects overfits, no hand help.

synth_engine's stress gate caught the max_list overfit AFTER synthesis, and the fix
was a human adding negative examples. This removes the human: the synthesizer itself
gates every candidate by stress-vs-oracle DURING the search, so it skips a program
that fits the examples but fails random cases and keeps looking for one that survives.

  for each candidate (in DSL order):
      fits the examples?  ->  AND survives 1000 stress cases vs oracle?  ->  return it
  the overfit (max_list init=0, fits all-positive examples) is auto-skipped; the
  search continues to init=first, which survives — with the SAME positive-only examples.

    python3 stress_synth.py
```

### structural_parser.py
```text
structural_parser.py — generalize over sentence STRUCTURE, not just words (the harder layer).

context_embed gave lexical generalization (paraphrase words). This gives STRUCTURAL
generalization: the same meaning across different sentence SHAPES, and compound/comparison
shapes the old keyword template ("X of Y") could never handle. It parses by ROLE
(entity/relation, found by slot not by position) and recognizes composite structures, then
PROPOSES a structured query the crisp solver VERIFIES — propose-and-verify, the membrane.

  single   : one entity + one relation, ANY phrasing  ("force of the rocket", "the
             rocket's force", "how much mass does the sample have")
  compound : >=2 relations, one entity            ("the mass and speed of the rocket")
  compare  : 2 entities + a relation + a compare word ("which has more force, rocket or sample")

Order-independent role extraction = robustness to phrasing; compound/compare = shapes beyond
any single template. The crisp side still answers, so a misparse that doesn't compute abstains.

Honest limit: still a small grammar of structural patterns (single/compound/compare), not a
full compositional grammar; nested clauses / conditionals are the next rung.
```

### student_trainer.py
```text
student_trainer.py — train the small LOCAL student on the distilled corpus.

The teacher (cloud model) produced varied phrasings -> Need labels in
distill_data.jsonl. This trains a lightweight student to map a NEW question to its
Need, generalizing beyond nl_query's lexical matcher via embeddings. Deployed it's
tiny, local, free, private — the big model never runs at inference.

Two student heads over GloVe sentence vectors (mean of content-word embeddings):
  * centroid — one mean vector per relation, predict by cosine (fast, crude).
  * kNN      — cosine-weighted vote over the k nearest TRAINING questions
               (stronger on small, diverse data; keeps every example).
No backprop, no GPU — the project's lightweight discipline. Held-out eval picks
the winner.

    st = Student.train(rows, glove)
    st.predict("how quick is the rocket?", method="knn")  -> ('attribute','rocket','speed')
```

### synth_engine.py
```text
synth_engine.py — one proposer-guided synthesis engine over all the spaces.

Unifies the scattered synthesizers (formula / accumulator-loop / two-state+conditional
/ while / early-return / list+nested / DP) behind ONE entry point. A task is a list of
(args, output) examples + an input kind; the engine ROUTES to the spaces that apply,
searches each (proposer-guided where built), and returns the first program that
VERIFIES on every example. Every backend emits a Python function; the engine verifies
uniformly by running it.

Then a BENCHMARK: a suite of classic algorithm tasks run through the engine, reporting
coverage (how many the brain writes, no LLM) and which space solved each.

    python3 synth_engine.py
```

### synth_invariant.py
```text
synth_invariant.py — self-made verifiers earn their keep in synthesis (invariant pre-filter).

Synthesis verifies a candidate with stress-vs-oracle: ~1000 oracle calls per candidate —
the expensive step. invariant_miner mints NECESSARY checks from the task's own examples for
nearly free. Put them in front of stress: a candidate whose outputs violate an admitted
invariant is rejected with NO oracle call. Only survivors pay for the full stress test.

  mine invariants from the task examples (validate on a few held-out oracle points, once)
  -> for each candidate: cheap invariant check; violate -> reject (0 oracle calls)
  -> survivors only -> expensive stress-vs-oracle

Necessary-not-sufficient (from invariant_miner) is exactly what a pre-filter wants: it can
only REJECT (cheaply), never falsely accept — so it never throws away a correct program,
it just spares the oracle from obviously-wrong ones.

    python3 synth_invariant.py
```

### synthesis_engine.py
```text
synthesis_engine.py — hardened Verifiable program synthesis (milestone #6).

Give it input/output examples; it searches a DSL (via the hardened search
engine) for a program that reproduces ALL of them, and returns it. The result
is correct BY CONSTRUCTION — the search goal is "matches every example" — so it
is verified, not guessed. It then generalizes to inputs it never saw, and fails
honestly (returns not-found) when no DSL program fits.

    se = SynthesisEngine()
    r = se.synthesize([("John Smith", "JOHN"), ("bob dylan", "BOB")])
    r.found, r.source          -> True, "upper -> first_word"
    r.apply("ada lovelace")    -> "ADA"   (generalizes; never shown)

Built on the hardened tree_reason search (optimal, deterministic): it returns
the SHORTEST correct program.
```

### template_memory.py
```text
template_memory.py — durable grammar store with the conjecture->verify->admit gate.

Mirrors PolicyMemory/PolicyLearner: induction PROPOSES a template (slotting / anti-unify),
verification against held-out labeled examples DISPOSES. Only verified templates are stored;
parsing is exact and explainable. Statement templates carry entity+value slots; question
templates carry an entity slot and answer with the rel. (Plan Phase A, Tasks 3/6.)
```

### test_open_lang.py
```text
test_open_lang.py — reject-tests for the open-language track (Gaps 1-4).
Standalone (no pytest). The point is the membrane: contradictions/type-violations must be
REJECTED, the genuinely-unknown must ABSTAIN, the good must ADMIT.
```

### test_phase_a.py
```text
test_phase_a.py — standalone tests for the Phase-A slice (no pytest dep).
Covers parse_template, template_memory, coverage_harness, domain_features, concept_memory.
Assertions follow the plan; slot_example stores NORMAL form (weighs->weigh) per Task 4.
```

### train_from_data.py
```text
train_from_data.py — train every faculty from ONE tagged corpus (brain_data format).

Pure-python faculties run anywhere (types, morphology, symbolic knowledge, predictor). The
student LM (torch/MPS) trains under venv2 with --lm. The C++ Brain perceives the raw text
with --brain. One file -> the whole brain, membrane intact (laws verified, not assumed).

    python3 train_from_data.py data/kimi_data.txt            # symbolic + predictor + mouth
    KMP_DUPLICATE_LIB_OK=TRUE venv2/bin/python3 train_from_data.py data/kimi_data.txt --lm --brain
```

### train_pipeline.py
```text
train_pipeline.py — PHASE 3 scaffold: the one unified neural training pipeline.

The insight (from the roadmap): the LM distillation, the SOM grounding, and the proposer
training are ONE pipeline — same corpus, same verified-solution signal. This is that driver.
It runs the three stages in a single pass:

  1. DISTILL  — corpus (optionally expanded by a qwen-coder teacher; falls back to given text)
  2. LM       — train the owned neural LM on the corpus (the probabilistic pillar)
  3. GROUND   — self-organize the SOM on data vectors + ground concepts (the fuzzy pillar)
  4. PROPOSER — learn synthesis-space priors from verified outcomes (the search guide)

It runs TINY here to prove the wiring end to end. The TRAINING PHASE is the same driver with
your resources: use_teacher=True (qwen-coder up), a real corpus, more epochs, a GPU. Nothing
else changes — the pipeline is already connected; only scale differs.

    python3 train_pipeline.py                    # tiny wiring proof (offline)
    venv2/bin/python3 train_pipeline.py --real   # + qwen-coder teacher + grounding on the SOM
```

### tree_domains.py
```text
tree_domains.py — same search engine, more kinds of reasoning.

Each domain below is a different KIND of problem — constraint satisfaction,
state-space puzzle, symbolic rewriting — yet all of them plug into the one
tree_reason.solve engine by supplying (operators, goal, heuristic). Nothing
about the search changes; only the rules do. That is the whole point: the
reasoning core is shared, each domain is a plug-in.

  N-Queens        constraint satisfaction (place queens, none attacking)
  Water jugs      state-space planning with a numeric goal
  Rewrite/proof   derive a target form from rules (a tiny formal system)
```

### tree_learn.py
```text
tree_learn.py — the search LEARNS to reason more efficiently from experience.

tree_reason searches with a hand-given heuristic (or none). This adds the
brain-like part: it watches itself solve easy instances, learns a heuristic
from that experience (a linear value over state features), and then expands
dramatically fewer states on new, harder instances. Reasoning that improves
with experience — lightweight, on CPU, no hand-tuned heuristic.

Domain: the 8-puzzle (large state space, where blind search is expensive and a
good heuristic matters enormously). The honest result is the node-count drop:

    blind search (no heuristic)   : thousands of states expanded
    LEARNED heuristic             : a few dozen — and it was learned, not coded

The learned weights end up close to the classic Manhattan heuristic — i.e. it
rediscovers a known-good heuristic from its own solved experience, with no one
telling it the rule.
```

### tree_reason.py
```text
tree_reason.py — the branch / prune / search reasoner.

The second reasoning engine for the project. The binding memory does ONE thing:
deductive closure over relations (A>B>C => A>C). This does the general thing
the scratchpad + PUCT were reaching for: explore a space of states by applying
rule-valid operators, prune the dead branches, and search to a goal — then read
back the path as the worked steps.

Transitive closure is just the simplest case of this (search whose only move is
"follow a relation"). The power comes from what the operators are. Define a
domain's operators + goal and the SAME engine solves it. Below: linear algebra
and the classic bridge-and-torch puzzle — two very different problems, one
search core, each showing its work.

This is honest about what it is: a clean general search (A* / uniform-cost),
not the untrained BG controller. The brain's learned critic could later supply
the heuristic; here the heuristic is hand-given per domain. Operators are
hand-defined per domain — that is the real, known limit (one domain at a time).
What's NOT limited: the solutions it finds inside those rules, including ones
you never handed it.
```

### type_oracle.py
```text
type_oracle.py — wires event_verify's injected `type_of` to a real taxonomy.

The membrane's selectional check needs a token's TYPE. The crisp source is the `isa` ladder
in `core_knowledge` (hand-checked, transitive): a token's type set is its full isa-closure
(dog -> mammal -> animal -> living_thing). Crisp beats fuzzy clustering here — exact and
explainable — and stays standalone (no C++/GloVe needed to run).

Honesty preserved: an UNKNOWN token returns None, so the membrane ABSTAINS (never guesses) —
exactly the three-valued contract. The `__call__` disposal path is CRISP-ONLY: fuzzy never
decides admit/reject (that would trade honest abstention for a guess). Instead an optional
`similar` hook (nearest-token from semantic_memory/context_embed) feeds `grow()`: fuzzy
CONJECTURES an isa edge -> a verify callback DISPOSES -> the edge is admitted into the crisp
closure. From then on the token disposes exactly, like any hand-checked isa fact. Fuzzy
narrows the teacher's work; it never crosses the membrane.

Usage:  oracle = TypeOracle();  admit(ev, store, oracle, constraints)
        oracle.grow("puppy", verify)   # conjecture->verify->admit into the taxonomy
(Open-language track — closes the 'type_of is a plug point' gap; fuzzy grows, crisp disposes.)
```

### validate.py
```text
validate.py — the promotion gate.

Basic infra for the prototype -> validate -> promote workflow. Run this before
moving any capability up the hierarchy: it exercises every validated mechanism
and checks the headline results still hold. Nothing gets promoted if this
regresses.

    python3 validate.py            # core correctness gate (a few minutes)
    python3 validate.py --full     # also run the slow performance sweeps

Each check runs a script and asserts a robust marker of its known-good result.
```

### verb_learn.py
```text
verb_learn.py — the capstone: LEARN a verb's selectional constraint from reading.

Positional detection recovers structure on an unknown verb, but the event ABSTAINS (held) —
we can't verify a verb we don't understand. This closes that: watch how a verb is used and
INDUCE its selectional restriction, so its events start disposing crisply (admit/reject).

The same conjecture->verify->admit membrane as type_oracle.grow, applied to verbs:
  * observe   — record the type-closures of each use's agent/patient (types from the noun
                oracle, which itself may have been grown from fuzzy neighbors).
  * conjecture — the constraint is the INTERSECTION of the observed role type-closures (the
                shared supertype), minus universal roots so it doesn't collapse to 'anything'.
  * verify    — a held-out use must satisfy the conjecture (no counterexample), AND the verb
                must have been seen >= promote_at times (a track record, not one sighting —
                like concept promotion; a learned constraint is a good hypothesis, still
                re-checked by the membrane on every future use, never a proof).

Admitting a verb makes it TRUSTED: its events move held -> admit/reject. (Open-language track
— read-only comprehension becomes learning-from-prose.)
```

### verifier_monitor.py
```text
verifier_monitor.py — verify the verifiers: detect irregular / untrustworthy checks.

Self-minted verifiers can go wrong: a SPURIOUS one rejects correct answers (worse than no
check — it blocks truth), a USELESS one never catches anything wrong (dead weight, false
confidence). The brain must audit its own checks, not just trust them. This is the meta
layer: monitor each verifier's behaviour against ground truth + against wrong candidates,
and flag the irregular ones for demotion or pruning.

  SPURIOUS  : rejects a KNOWN-CORRECT output            -> demote (it blocks truth)
  USELESS   : never rejects ANY wrong candidate         -> prune (no discriminative power)
  HEALTHY   : catches wrong ones, never rejects correct -> keep

This closes the loop on self-made verifiers: mine -> validate -> admit -> USE -> AUDIT ->
demote/prune. A verifier only keeps its authority while it behaves.
```

### whole_brain.py
```text
whole_brain.py — one front over the whole system. The pieces, made whole.

A single ask(text) routes a request to the right faculty and returns a verified
answer with provenance:

  COMPUTE   "force of the rocket"        -> means-ends executive over facts+policies
  FACTUAL   "is a dog a mammal" / "what  -> ReasoningEngine over real knowledge
             can a bird do"                 (transitive isa + property inheritance)
  CODE      "write a factorial function" -> synth_engine (verified) or recall from store
  UNKNOWN   anything else                -> honest "I don't know"

Discovered/synthesized knowledge PERSISTS in the BrainStore — ask for the same
function twice and the second time it's recalled, not re-synthesized. Every answer
carries how it was produced and whether it's verified.

    python3 whole_brain.py
```

### word_math.py
```text
word_math.py — controlled arithmetic word problems (language -> operation).

The brain already computes (the search engine solves algebra). The hard part was
never the math — it's mapping a sentence to the operation. This is a grammar
extractor: pull the numbers, detect the operator from a verb (give/lose ->
subtract, get/buy -> add), then compute and show the arithmetic.

    solve("I have 10 apples and give 3 away, how many do I have left?")
        -> "7 apples. (10 - 3 = 7)"

Honest scope: controlled single-operation problems (add / subtract) over a
running quantity. Multi-step, rates, ratios, or free phrasing need a real
semantic parser (an LLM). Returns None when it can't parse one cleanly.
```

### world_knowledge.py
```text
world_knowledge.py — load basic human knowledge of the world (ConceptNet).

ConceptNet 5.7 is a crowd-built common-sense knowledge graph — already
(subject, relation, object) triples, which is exactly what the binding memory
eats. This extracts a CURATED English subset (high-weight, common relations)
into the brain's relation vocabulary, and caches it so re-runs are instant.

Honest scope: ConceptNet is real common sense but noisy and incomplete; this is
the "feed it curated, trusted knowledge" path, not "crawl the web." A few
thousand high-weight assertions make a responsive, demonstrable world model.

    facts = load_conceptnet(max_facts=6000)   # [(subj, rel, obj), ...]
```

