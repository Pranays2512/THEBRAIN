# brain2 architecture

132 modules that used to sit flat in one directory are now grouped into **5 layered
packages**. Dependencies point downward (a package imports only from packages below
it), with three documented exceptions.

```
tests        the acceptance + unit suites; may import anything below
  │
training     train_all, train_pipeline, train_from_data, read_pdf_train,
  │          knowledge_distill, student_trainer, brain_data
faculties    orchestrators + standing-loop drives: whole_brain, read_book,
  │          reading_loop, conversation_engine, query_planner, neuro-driven
  │          agents, curiosity/appraisal/autonomous loops
adapters     IO edge (depends on engines): llm_adapter, llm_extractor, mouth,
  │          ocr_pdf, nl_front, chat, brain_repl, server, math_chat
engines      pure Python engines — the foundation
             ├── reasoning   reasoning_engine, tree_reason, means_ends,
             │               nested_parser, structural_parser, neuro_bridge, dual_process
             ├── synthesis   synth_engine, program_synth*, loop_synth*, dp_*,
             │               composable_*, proposers, inductive/policy induction
             ├── math        algebra/calculus/integral/physics/word_math,
             │               math_parser, factorizer, dimensional_verify, prob_compute
             ├── knowledge   knowledge_engine/base/pack, concept_*, semantic_memory,
             │               world_knowledge, fact_extractor
             ├── grounding   ground_*, grounding, crispify_bridge, context_embed
             ├── events      event_form/parse/verify, verb_learn, discourse, analogy_*
             ├── neural      neural_lm, neural_lm_torch, cpp_accel
             └── store       brain_store, check_library, template_memory, type_oracle,
                             parse_template, corpus_scale, coverage_harness
```

## The native C++ core

`core/` holds the **C++ engine source** (`*.hpp`, `*.cu`, `*.cuh`). Together with
`brain2.cpp` at the repo root and `CMakeLists.txt`, it compiles into the native module
`brain2.cpython-*.so`, imported from Python as `import brain2`. The C++ `core/` is not
part of the Python package tree and is not imported with `from core...`.

## Rules

- **Import direction is downward.** Cross-package imports use the full path,
  e.g. `from engines.reasoning.reasoning_engine import ReasoningEngine`,
  `from adapters.mouth import Mouth`. A module never imports from a package above its
  own layer. (Package `__init__.py` facades were evaluated but not adopted — populating
  them triggered import-time circular loading; full-path imports are the interface.)
- **Run from `brain2/`.** The current working directory is on `sys.path`, so the
  absolute package imports resolve. Scripts are invoked either by their root shim
  (`python3 train_all.py`) or as a module (`python3 -m tests.harden_regress`).
- **Root shims** (`*.py` at the top level) are one-line `runpy` launchers kept only
  for entrypoints that are actually run by hand or by the settings allow-list. They
  preserve every existing command; the real code lives in the packages.

## Documented layer exceptions

Two imports point upward and are kept as-is (resolving them would require changing
call structure, which this refactor does not do). See `_refactor/exceptions.md`.

- `engines/knowledge/fact_extractor` → `faculties/conversation_engine`
  (`fact_extractor` is itself imported by `engines/knowledge`, so it cannot move up).
- `adapters/nl_front` → `training/student_trainer`
  (`nl_front` invokes the student trainer from its front-end path).
- `engines/reasoning/neuro_bridge` → `faculties/conversation_engine`, `faculties/query_planner`
  (a middle-of-stack bridge: imported by engines modules yet wires up two faculties —
  no placement removes the up-edge).

## Verification

`_refactor/baseline/` holds the pre-refactor output of 6 acceptance suites (`exam`,
`test_open_lang`, `test_phase_a`, `stress_exam`, `reasoning_suite`,
`component_validation`). The restructure was performed one package at a time; after
every move `_refactor/gate.py` re-ran all six and required byte-identical output, so
runtime behavior is provably unchanged.

