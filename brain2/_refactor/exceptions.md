# Documented up-edge exceptions

These imports point "up" the layer order (engines → adapters → faculties → training →
tests) and are kept as-is (no logic change). They are the edges not resolved by module
placement.

- `adapters/nl_front → training/student_trainer`
  nl_front's front-end path invokes the student trainer directly. Relocating nl_front
  would not remove the up-edge (it would still reach training), so it stays documented.
- `engines/knowledge/fact_extractor → faculties/conversation_engine`
  fact_extractor pulls a helper from conversation_engine (a lazy in-function import).
  Kept as a documented exception rather than rewiring call structure.
- `engines/reasoning/neuro_bridge → faculties/conversation_engine, faculties/query_planner`
  neuro_bridge is a genuine middle-of-the-stack bridge: it is imported by engines
  modules (so it must sit low) yet itself wires up two faculties (so it reaches high).
  No single placement removes the up-edge; kept documented.
