# Brain2 — Architecture Flaws & Limits (audit + triage)

A deep, code-level audit of every layer, with an honest triage of what's safely
fixable now, what's a big redesign, and what's a *deliberate* tradeoff (not a bug).

---

## Triage summary

**Fix-now (safe, low-risk, real) — ALL DONE:**
- ~~`think()` fake `coherence = 1.0`~~ → real mean-cosine of inner-speech words. **DONE.**
- ~~Debug `printf` scattered / not thread-safe~~ → `core/debug.hpp` `B2DEBUG`, env-gated
  (`BRAIN2_DEBUG`); all sites routed through it. **DONE.**
- ~~Static non-thread-safe instrumentation counters~~ → `std::atomic<int>`. **DONE.**
- ~~`DecoderRNN::save()` omits `lr_`~~ → persisted (trailing, backward-compatible). **DONE.**
- ~~`teach_fact` no gate~~ → conflicting overwrites counted (`Brain::crisp_conflicts`) +
  logged instead of silent; exposed to Python. (Single-scalar store unchanged.) **DONE.**

**Big redesign (real, high value, multi-session, risk to a working core):**
- `DecoderRNN` has no training loop (vanilla Elman, random weights forever) → implement BPTT.
- `daydream()` predictor update is commented out (needs MDN 128-D coords) → re-enable.
- Predictive-coding error doesn't flow back into SOM learning → couple the signals.
- Working-memory `context()` is a flat average → destroys combinatorial structure for
  multi-hop reasoning; needs structured/slot-preserving aggregation.
- BG controller has no temporal abstraction (flat 31 ops) → macro-ops / sub-goals.
- Symbolic store is an island (BG only touches the scratchpad) → couple them.
- `brain.hpp` is a 1,425-line god object, all components public → modularize.
- Episodic triple-gate freezes storage at convergence → novelty/min-rate store.
- Prioritized replay (currently uniform, bounded 300) → priority + larger/paged buffer.
- SOM neuron count fixed at construction → grow on capacity pressure.
- `crisp_facts` has no uncertainty/provenance → add confidence + conflict handling.
- Binding-memory LSH 8-bit/256 buckets at 50k cap (~195/bucket) → more planes.
- SOM activation map near-binary (1-hot + 16 neighbors) → richer activation.

**Deliberate tradeoffs (NOT bugs — chosen on purpose):**
- SOM `find_bmu` brute-force O(N): "correctness first" — the NSW/LSH path is approximate
  and corrupts SOM training. Re-enabling trades exactness for speed; only past ~1M neurons.
- Verifiability ceiling (open/creative/unverifiable → LLM): the design's defining bound.
- Bounded replay window: a known forgetting/compute tradeoff.

---

## 1. Scalability Walls

### SOM is Brute-Force O(N) per Token
`som.hpp` find_bmu is labeled *"Fast O(log N) Greedy Graph Search -- SWAPPED TO BRUTE
FORCE Linear Scan (Correctness first)"*. The NSW graph (`neighbors_`, Hebbian rewiring)
is maintained at O(N) write cost but never used for search — paying NSW maintenance,
getting no O(log N) query benefit. **Tradeoff (correctness-first), not a bug; revisit
only past ~1M neurons.**

### SOM Activation Map is Nearly Binary
`activation_map` sets `acts[bmu]=1.0` then only the Hamming-1 neighbors decay — ~17
nonzeros / 4096. Downstream PC/WM/episodic get a near-binary, low-information signal.

### Binding Memory LSH is Only 8-Bit / 256 Buckets
`N_PLANES=8` → 256 buckets; at 50k cap that's ~195 triples/bucket. Exact recall degrades
as count nears capacity. (More planes = a real fix.)

---

## 2. Architectural Coupling & Integration

### brain.hpp Is a 1,425-Line God Object
Owns all 24 components as public members; training scripts reach into `brain.som`/
`.predictor`/… bypassing perceive/think/speak; shared/partial mutex paths risk races.

### The Fuzzy↔Crisp Membrane Is Asymmetric and Partial
The confidence gate lives in `crispify_bridge.py` (Python) and is **not enforced in C++**.
`crisp_facts` is a plain `std::map` with no confidence/decay; `teach_fact` writes with no
gate — any Python caller can poison the crisp store. **High; the fix-now item is adding a
gate/confidence to teach_fact.**

### Predictive-Coding Layers Don't Feed Back Into SOM Learning
The 4 PC layers compute errors that never update SOM weights — the SOM trains only via
Hebbian/competitive learning, decoupling the two signals (defeats PC's premise).

### `daydream()` Doesn't Train the Predictor
The predictor weight update is commented out (`// predictor.step(...)`, needs MDN 128-D
coords). The validated consolidation benefit comes from `dream_replay_faithful`, not
`daydream`. **High.**

---

## 3. Language & Grounding

### Word Embeddings Restricted to Half the Brain
New word vectors get nonzero components only in dims `0..n_dims/2` ("left hemisphere").
Perceptual concepts span all dims → systematic low cosine bias for grounding queries.

### DecoderRNN Is a Vanilla Elman RNN With No Training Loop
No LSTM gates, no BPTT, no `train()` — only `generate()`/`save()`/`load()`. Initialized in
Brain, never trained → random weights forever despite being documented as the "generative
sequence decoder". **High.**

### `think()` Hard-Codes `coherence = 1.0`
`result.coherence = 1.0f;` regardless of trajectory quality — any downstream quality gate
gets a meaningless constant. **Fix-now.**

### OOV Words Auto-Registered With Random Embeddings
`perceive_text` calls `register_word(w)` for unknowns → random small vector that
cosine-matches existing words by chance → false grounding; can't distinguish "unknown"
from "random embedding".

---

## 4. Memory System

### Episodic Commits Triple-Gated → Freeze at Convergence
`episodic_active && (ce > ep_thr) && commit(ce)`: a converged (low-CE) model rarely fires
`ce > ep_thr` → stops storing exactly when it meets novelty. **High.**

### Replay Buffer Uniform Random (No Priority), Bounded 300
Rare/surprising events replayed at 1/N like mundane ones; >300 sequences → older
permanently discarded → asymptotic forgetting on an unbounded stream. **High.**

### Working Memory Context Is a Flat Vector Average
`context()` blends all slots into one vector; multiple WM concepts (multi-hop reasoning)
are averaged together, destroying combinatorial structure. **High.**

---

## 5. Reasoning & Search

### BG Controller Has No Temporal Abstraction
2-layer MLP over 31 atomic ops; no macro-ops/sub-goals/hierarchy → flat primitive
sequences, brittle multi-step reasoning, slow TD(λ) learning. **High.**

### `reason()` PUCT `c_puct = 2.0` Hardcoded / Unswept
No tuning/schedule; with a cold critic exploration is suppressed, with a warm one the
untrained prior dominates.

### The Symbolic Layer Is an Island
BG selects ops over the *scratchpad*, never reads/writes `Symbolic` → neurosymbolic
integration mediated by flat vector slots, losing structure/relations. **High.**

---

## 6. Engineering & Code Quality

- **Static instrumentation counters not thread-safe** (`static int` in `perceive` /
  `train_lm_sequence_fused`) → data races under parallel Python calls. **Fix-now.**
- **Debug `printf` everywhere**, no logging framework, print-first-N via `static int`
  (not thread-safe) → pollutes prod output. **Fix-now (guard).**
- **`load_components` rebuilds deps without validation** → loaded vs constructor-random
  components mismatched in one checkpoint.
- **`DecoderRNN::save()` omits `lr_`** → fresh 0.01 LR on resume → possible spike. **Fix-now.**

---

## 7. Conceptual / Design Limits

- **Verifiability ceiling** — open/creative/probabilistic/perceptual-inference → LLM;
  ~18% of language queries fall through the 82% student to the (stateful, nondeterministic)
  LLM. *Deliberate design bound.*
- **No true continual learning** — only replay, bounded at 300 → asymptotic forgetting on
  an unbounded task stream. *Tradeoff; needs unbounded/paged + priority replay.*
- **SOM topology fixed at construction** — `expand_dims` grows dimensions, not neurons →
  runs out of representational capacity as knowledge grows. **High.**
- **No uncertainty in the crisp store** — `crisp_facts` = single scalars, no bounds/
  provenance/decay → conflicting values (boiling point at altitude) only overwrite.

---

## Severity table

| Layer | Issue | Severity | Class |
|---|---|---|---|
| SOM | O(N) brute force; graph unused | High | tradeoff |
| SOM activation | near-binary signal | Medium | big |
| LSH | 8-bit/256 buckets at 50k | Medium | big |
| Brain class | god object | Medium | big |
| Crisp↔Fuzzy | no C++ gate; teach_fact bypass | High | fix-now (gate) |
| PC | error doesn't reach SOM | Medium | big |
| Daydream | predictor update commented | High | big |
| Lang embeddings | half-space cosine bias | Medium | big |
| DecoderRNN | no training loop | High | big |
| think() coherence | always 1.0 | Low | fix-now |
| OOV | random vectors silently | Medium | big |
| Episodic | triple gate freezes at convergence | High | big |
| Replay | uniform, bounded 300 | Medium | big |
| WM context | flat average | High | big |
| BG | flat ops, no abstraction | High | big |
| PUCT | c_puct=2.0 unswept | Medium | big |
| Symbolic↔BG | no coupling | High | big |
| Instrumentation | static races | Medium | fix-now |
| Debug prints | no logging | Low | fix-now |
| Checkpoint load | mismatched states | Medium | big |
| DecoderRNN lr_ | not persisted | Low | fix-now |
| Continual learning | bounded replay | High | tradeoff/big |
| SOM capacity | fixed neuron count | High | big |
| Crisp uncertainty | scalar facts only | Medium | big |
