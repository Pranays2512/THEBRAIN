# Decouple brain2 — behavior-preserving package restructure

**Date:** 2026-07-09
**Branch:** grounded-math
**Checkpoint commit:** a043d28 (pre-refactor restore point)

## Problem

`brain2/` is 132 Python files flat in one directory. The dependency graph is
unreadable — 132 peers in a single namespace, no packages, no layers, no declared
direction. `whole_brain` imports ~a third of the tree (fan-out 38); a handful of
engines (`means_ends` fan-in 16, `tree_reason` 14, `synth_engine`/`reasoning_engine`
12) are load-bearing but their role isn't visible from the structure.

**Goal:** a clean, layered, understandable architecture — proper routing and
decoupling — **without changing any logic, execution flow, or the working of the
brain.** The runtime behavior must be byte-for-byte identical before and after.

## Hard constraints

1. **Zero behavior change.** No logic edits, no dependency inversion, no new
   interfaces or indirection. The only permitted changes are: moving files into
   packages, rewriting import statements to the new paths, and adding pure
   re-export `__init__.py` facades.
2. **Proven, not asserted.** A full acceptance gate (below) runs green before the
   refactor and must produce identical output after every single package move.
3. **Invocation unchanged.** Every existing run command and the settings.json
   permission allow-list keep working.

## Enabling fact

The import graph has **zero cycles**. A topological layering therefore exists, so
the packages can be a *true partition of the actual dependency DAG* — the layering
is **descriptive** (derived from real edges), never imposed by rewiring.

## Target architecture

Dependency direction (bottom = foundation, high fan-in): **core → io → faculties →
training → tests**. `experimental/` sits off to the side; `cpp/` holds the native
build.

```
brain2/
  core/                 # pure engines — foundation
    reasoning/          # reasoning_engine, tree_reason, means_ends, nested_parser, dual_process...
    synthesis/          # synth_engine, program_synth*, loop_synth*, dp_*, composable_*, proposers
    math/               # algebra/calculus/integral/physics/word_math, factorizer, dimensional_verify
    knowledge/          # knowledge_engine/base/pack, concept_*, semantic_memory, world_knowledge
    grounding/          # ground_*, grounding, crispify_bridge, context_embed, domain_features
    events/             # event_form/parse/verify, verb_learn, discourse, analogy_*
    neural/             # neural_lm, neural_lm_torch, cpp_accel
    store/              # brain_store, check_library, concept_memory, template_memory, type_oracle
  io/                   # adapters (depend on core): llm_adapter, mouth, ocr_pdf, nl_front,
                        #   structural_parser, server, chat, brain_repl
  faculties/            # orchestrators: whole_brain, read_book, reading_loop,
                        #   conversation_engine, query_planner, neuro_bridge
  training/             # train_all, train_pipeline, train_from_data, read_pdf_train,
                        #   knowledge_distill, student_trainer
  tests/                # harden_regress, exam*, test_*, stress_*, validate,
                        #   component_validation, reasoning_suite
  experimental/         # one-off probes, scratch, dead scripts — quarantined, labeled
  cpp/                  # brain2 binding source + build.sh (the .so stays importable)
```

Each package and each `core/` subdomain gets an `__init__.py` facade exposing its
public names, so cross-package code reads `from core.reasoning import
ReasoningEngine`. The graph collapses from 132 peers to ~6 top-level nodes.

> The per-file assignment above is a **draft from filenames**. The first
> implementation step produces the authoritative `old_module → new_path` map by an
> automated classification pass, reviewed by the user before anything moves.

## Behavior-preservation mechanics

- **Facades are pure re-export.** `__init__.py` contains only
  `from .reasoning_engine import ReasoningEngine` lines — no logic, no runtime
  indirection. The same module objects load.
- **Run from `brain2/` as today** (cwd on `sys.path`). Imports become absolute:
  `from core.reasoning import ReasoningEngine`. No `brain2`-as-package, no `-m`.
- **Codemod, not hand-editing.** A script builds the `old_module → new_path` map and
  rewrites every import site with libcst/AST (not regex), one deterministic pass per
  package.
- **Invocation stays identical via root shims.** Every directly-invoked entrypoint
  (`train_all.py`, `train_pipeline.py`, `read_pdf_train.py`, `harden_regress.py`,
  `exam.py`, ...) keeps a 1-line shim at `brain2/` root:
  `from training.train_all import main; main()`. Existing commands and the
  allow-list keep working; implementation moves, the front door doesn't.

## Acceptance gate

Candidate suites: `harden_regress`, `exam`, `test_open_lang`, `test_phase_a`,
`stress_exam`, `reasoning_suite`, `component_validation`, `validate`.

1. **Green + determinism screen.** Run each candidate suite **twice** before
   touching anything. Keep only the ones that pass today *and* produce byte-stable
   output; timestamp/RNG-noisy suites get a normalizer (strip the varying line) or
   are dropped from the gate with a note. Suites that don't pass on the clean
   checkpoint are excluded and noted (they can't prove preservation of behavior they
   don't exercise cleanly).
2. **Baseline capture.** Snapshot each stable suite's stdout to `baseline/*.out`.
3. **Per-move verification.** After every package move: run codemod → run the full
   gate → `diff baseline/X.out <(python X.py)` must be empty for all X. Green =
   commit that package. Any diff = revert that one move and investigate.

## Migration order

Bottom-up, one package per commit (~13 green checkpoints):

`core/store → core/neural → core/math → core/events → core/grounding →
core/knowledge → core/reasoning → core/synthesis → io → faculties → training →
tests → experimental`

## Edge, quarantine, and C++ handling

- **Up-pointing edges (~20).** Genuine coupling smells (e.g. `knowledge_base →
  neuro_bridge`, `brain_data → knowledge_distill`, and mislabeled ones like `exam`
  which is really a test). Each is resolved by **relocating the module to the layer
  its real edges point to** — never a code rewrite. A small number may remain as
  **documented exceptions**.
- **`experimental/` quarantine.** Auto-candidates = files with zero inbound edges
  from core/io/faculties/training and not an entrypoint. The user approves the
  quarantine list before anything moves.
- **C++.** `import brain2` must keep resolving; the compiled `.so` stays where the
  import finds it. Only `build.sh` and the binding *source* move to `cpp/`. The gate
  (many suites load the native brain) proves the import still resolves.

## Out of scope

- Any change to brain logic, algorithms, training, or runtime behavior.
- Dependency inversion / introducing abstract interfaces.
- The root-level untracked items unrelated to brain2 (`Brain/`, `Dockerfile`,
  `.agents/`, `extract_*.py`, scratch docs).
- Regenerating or altering the graphify code-graph output.

## Success criteria

- The `brain2/` top level reads as ~6 labeled packages, not 132 flat files.
- Every acceptance suite produces output identical to the pre-refactor baseline.
- Every existing run command and the settings.json allow-list still work.
- Cross-package dependencies go through package facades; the layer direction
  (core → io → faculties → training → tests) holds, with any exceptions documented.
