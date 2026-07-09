# Documented up-edge exceptions

These two imports point "up" the layer order (core → io → faculties → training →
tests) and are kept as-is (no logic change). They are the only edges not resolved
by module placement.

- `nl_front (io) → student_trainer (training)`
  nl_front's front-end path invokes the student trainer directly. Relocating nl_front
  would not remove the up-edge (it would still reach training), so it stays documented.
- `fact_extractor (core/knowledge) → conversation_engine (faculties)`
  fact_extractor pulls a helper from conversation_engine. Kept as a documented
  exception rather than rewiring call structure.
