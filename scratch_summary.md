### analogy.hpp
```text

```

### attention.hpp
```text
/*
 * attention.hpp — Attention System, Component 7 of Brain v2
 *
 * Attention is a gating filter between perception and working memory.
 * Not a transformer — biological spotlight attention:
 *   - Maintains a saliency map over SOM neurons
 *   - Novelty (prediction error) increases saliency at active neurons
 *   - Emotion modulates threshold: high arousal = narrower focus
 *   - Decay makes attention drift if no new signals arrive
 *
 * Output: attentional weights (0–1 per neuron) for any downstream consumer.
 * gate() decides whether a given activation passes to working memory.
 *
 * Two mechanisms:
 *   1. Bottom-up (stimulus-driven): novelty / prediction error → saliency spike
 *   2. Top-down (goal-driven): external bias signal boosts specific neurons
 *
 * Saliency update (per active neuron i):
 *   saliency[i] = saliency[i] * (1 - decay) + novelty * activation[i]
 *
 * Gate threshold:
 *   base_threshold * (0.5 + 0.5 * arousal_modulator)
 *   High arousal → higher threshold → only strongest signals pass
 */
```

### basal_ganglia.hpp
```text
// Operations the BG controller can select
```

### binding_memory.hpp
```text
// ────────────────────────────────────────────────────────────
// LSH Index: 8 random hyperplanes → 2^8 = 256 buckets.
// Reduces average query scan from O(n) to O(n/128) without changing the API.
// ────────────────────────────────────────────────────────────
```

### brain.hpp
```text
/*
 * brain.hpp — Integration Layer, Brain v3
 *
 * Wires all 16 components into a unified cognitive loop:
 *
 * PERCEIVE (external input → SOM → Predictor → error → all components):
 *   1. SOM: raw vector → activation map + BMU
 *   2. Predictor: predict next activation (online, weight update)
 *   3. Attention: gate activation by novelty + arousal
 *   4. Emotion: update from prediction error
 *   5. WorkingMemory: gate if attention passed
 *   6. EpisodicMemory: observe + commit if surprising
 *   7. SelfModel: observe internal state
 *
 * THINK (inner speech — runs N steps without external input):
 *   1. WorkingMemory context → Language decode → best word
 *   2. Re-encode word → Predictor step (offline)
 *   3. Imagination step: predict next concept
 *   4. Push predicted concept back into WorkingMemory
 *   5. Repeat
 *
 * SPEAK (concept sequence → word sequence):
 *   Language.speak(concept_sequence)
 *
 * DREAM (rest-phase consolidation):
 *   1. Episodic retrieve top memories → seeds
 *   2. Imagination dream from seeds
 *   3. Extract high-coherence frames → WorkingMemory
 *   4. Consolidate episodic memories
 */
```

### debug.hpp
```text
// Tiny debug-logging guard. Production code had scattered printf("DEBUG...") /
// printf("[...]") with non-thread-safe `static int print_count` gates that polluted
// stdout and couldn't be silenced without editing source. Route them through
// B2DEBUG(...) — silent unless the env var BRAIN2_DEBUG is set (checked once).
```

### decoder.hpp
```text
// A lightweight Recurrent Neural Network (Elman RNN) for Sequence-to-Sequence generation
```

### emotion.hpp
```text
/*
 * emotion.hpp — Emotion System, Component 6 of Brain v2
 *
 * Valence   [-1, +1]: negative (fear/pain) → positive (reward/pleasure)
 * Arousal   [ 0,  1]: calm (sleep) → excited (panic/joy)
 *
 * Emotion is NOT a separate module — it's a global modulation signal:
 *   - High arousal → boosts attention threshold (more selective)
 *   - Positive valence + high arousal → exploration mode
 *   - Negative valence + high arousal → avoidance mode
 *   - Emotion modulates learning rate (surprise * arousal amplifier)
 *   - Salience for working memory gating = arousal * abs(valence)
 *
 * Learning:
 *   Emotion state drifts toward "emotional input" (valenced prediction error).
 *   decay_rate moves back toward neutral when no signal.
 *   Strong events leave emotional traces that decay slowly.
 *
 * Trigger sources:
 *   - Prediction error (surprise = arousal increase)
 *   - Goal proximity (positive valence)
 *   - Threat signals (negative valence, high arousal)
 *   - Internal needs (hunger/fatigue-like drives — abstracted as need_level)
 */
```

### episodic.hpp
```text
/*
 * episodic.hpp — Hierarchical Spiking Episodic Memory (Hippocampus)
 *
 * Episodes are no longer flat arrays, but hierarchical trees (Roots -> Chunks -> Frames).
 * This structure enables extremely fast O(log N) retrieval across large narrative sequences.
 */
```

### factorizer.hpp
```text
// factorizer.hpp — native port of factorizer.eval_tree (the hot path that verifies factored
// expressions evaluate identically). Python serialises an expression tree to an S-expression
// ("(* (+ a b) c)"); C++ parses + evaluates it under a variable environment. Operators + - * /,
// leaves are numbers or variable names. Used to check meaning is preserved under factoring.
```

### global_workspace.hpp
```text
// Integer IDs for modules that can bid for the global broadcast
```

### hierarchical_predictor.hpp
```text
/*
 * hierarchical_predictor.hpp — Multi-timescale LSTM predictor (Brain V3)
 *
 * Three nested levels:
 *   Level 0 (fast)   — 1-step: delegates to existing Predictor in brain.hpp
 *   Level 1 (chunk)  — N-step summaries: learns rules over reasoning chains
 *   Level 2 (episode)— M-chunk summaries: learns episode-level structure
 *
 * Only levels 1 and 2 are defined here; level 0 is the existing Predictor.
 * Error at each level is LOCAL — does not propagate down.
 */
```

### imagination.hpp
```text
/*
 * imagination.hpp — Imagination (Offline Simulation), Component 5 of Brain v2
 *
 * Runs the Predictor in offline mode — simulates sequences of activations
 * that haven't happened yet. Same prediction engine as online processing,
 * but disconnected from real input and no weight updates.
 *
 * "What if?" — brain feeds hypothetical start state → simulates N steps forward
 * → evaluates outcome (good/bad) → informs decisions without real-world cost.
 *
 * Dreams = imagination running during rest with random or memory-seeded starts.
 *
 * Output: sequence of predicted activation vectors + quality score
 * Quality = mean similarity between consecutive frames (coherence measure)
 */
```

### invariants.hpp
```text
// invariants.hpp — native port of the proven invariant checker (invariant_miner.py).
// Self-minted necessary-condition verifiers: mine the invariants that hold across examples,
// then reject any candidate output that violates one — a fast pre-filter before the
// expensive oracle. Stable + verified in Python; ported here for the synthesis hot path.
```

### language.hpp
```text
/*
 * language.hpp — Bidirectional Language, Component 8 of Brain v2
 *
 * Words are learned SOM vectors — same concept space as perception.
 * No hardcoded grammar. No template rules.
 * Grammar emerges from sequence statistics learned in the Predictor.
 *
 * Encoding (word → concept vector):
 *   lookup table: word string → float[n_dims]
 *   Learned: each word's vector drifts toward co-occurring concept activations
 *
 * Decoding (concept vector → word):
 *   nearest-neighbor search in word embedding table
 *   Returns top-k candidate words with similarity scores
 *
 * Inner speech:
 *   WorkingMemory context → decode → emit word → re-encode → observe in SOM
 *   This loop runs autonomously and is what generates thoughts.
 *
 * Learning:
 *   When word heard AND SOM activation present:
 *     word_vec += lr * (som_activation - word_vec)
 *   Word vectors drift toward what the brain perceives when hearing them.
 */
```

### logic_engine.hpp
```text

```

### lsh.hpp
```text
/*
 * Cognitive TLB (Translation Lookaside Buffer) using Locality-Sensitive Hashing (LSH).
 * Maps a continuous "logical" thought vector to a "physical" BMU address in O(1) time.
 * Mimics an OS Page Table for the Neural Architecture.
 */
```

### memoization.hpp
```text
// Global Memoization Cache for the Brain
```

### policy_engine.hpp
```text
// policy_engine.hpp — the CRISP symbolic reasoner, in C++.
//
// The C++ port of the Python neurosymbolic executive (means_ends.py +
// policy_proposer.py). It lives BESIDE the vector-symbolic BindingMemory as the
// crisp half of the brain: discrete string symbols, exact tuple-formula
// evaluation, verified policy learning — the membrane preserved in C++.
//
//   * PolicyMemory  — stored composition rules (target = f(inputs)), separate
//                     from the fuzzy binding memory.
//   * PolicyEngine  — means-ends solver with memoization (tabling) + cycle guard
//                     + a groundability PROPOSER that orders candidate policies,
//                     and a conjecture -> verify -> admit loop for learning new
//                     policies by composition (numeric gate; no fuzzy leakage).
//
// Facts come through a callback (std::function), so the engine can be fed by the
// Python KB or, later, a crispified view of the C++ BindingMemory. Header-only,
// no external dependencies.
```

### predictive_coding.hpp
```text

```

### predictor.hpp
```text
/*
 * predictor.hpp — LSTM Prediction Engine, Brain v2
 *
 * FIXES APPLIED:
 *   LM Head implementation with Weight Tying against GloVe embeddings.
 *   Softmax cross-entropy over restricted corpus vocabulary.
 */
```

### procedural_memory.hpp
```text

```

### proposer.hpp
```text
// proposer.hpp — native port of the learned proposer's scoring (feature_learner.py):
// disc_weights (a feature's weight = its variance across the winning-space profiles, i.e.
// how well it SEPARATES spaces) and feat_sim (weighted similarity of a task's features to a
// space profile). This is the hot scoring the proposer runs to order synthesis spaces.
```

### reasoning.hpp
```text
/*
 * reasoning.hpp — Reasoning Engine, Component 12 of Brain v2
 *
 * System 2 thinking: slow, deliberate, step-by-step.
 * Works with Scratchpad + Symbolic to chain operations.
 *
 * Complements the fast pattern-matching brain (System 1).
 *
 * A ReasoningStep = {input_slot, op_symbol, arg_slot, output_slot}
 * Chain of steps = a reasoning program.
 *
 * Example: solve "a + b = ?"
 *   step 0: read("a"), apply("+"), read("b") → write("result")
 *
 * Example: proof chain "if A then B, if B then C → A implies C"
 *   step 0: read("A"), apply("->"), read("B")  → write("step1")
 *   step 1: read("step1"), apply("->"), read("C") → write("conclusion")
 *
 * Convergence: stop when output slot stops changing (delta < threshold)
 * or max_steps reached.
 *
 * Also supports:
 *   - Unary ops (negate, copy)
 *   - Conditional: only execute step if similarity > threshold
 *   - Loop: repeat a sub-chain until convergence
 */
```

### reasoning_ops.hpp
```text
// reasoning_ops.hpp — the last two proven primitives ported to C++, closing phase 2:
//   cosine_map    : cosine similarity of two sparse feature maps (context_embed.cosine)
//   analogy_score : corresponded-relations count under an entity mapping, with a consistent
//                   relation map (analogy_struct._score) — structure-mapping's core.
```

### refuter.hpp
```text
// refuter.hpp — native port of the refuter's core (refuter.py): given a candidate's outputs
// and the oracle's outputs over a contiguous integer input range, find WHERE the candidate
// breaks and characterise the VALID SCOPE (the integer intervals where it holds). The
// verifier turned aggressive — hunts disagreement, not agreement. Hot path; proven in Python.
```

### regularity.hpp
```text
// regularity.hpp — native port of irregularity_detector._law_error. Fits a linear and a
// power law to training (x,y) points by least squares and returns the best held-out relative
// error. This is the math behind "does this domain have a checkable regularity" — the brain's
// map of where verification can reach. Pure numeric, deterministic, proven in Python.
```

### scratchpad.hpp
```text
/*
 * scratchpad.hpp — Scratchpad Memory, Component 11 of Brain v2
 *
 * External memory tape for deliberate, step-by-step reasoning.
 * Bypasses Working Memory's 7-slot limit entirely.
 *
 * Like paper for a human doing math:
 *   - Write intermediate results to named slots
 *   - Read them back in later steps
 *   - No decay, no capacity limit, no interference with WM
 *
 * Also supports:
 *   - Stack (push/pop) for recursive reasoning
 *   - History per slot (last N writes) for backtracking
 *   - Slot tagging (what kind of value is stored here)
 *   - Diff: compare two slots (are they similar?)
 */
```

### self_model.hpp
```text
/*
 * self_model.hpp — Self-Model, Component 9 of Brain v2
 *
 * Observes brain's own internal state — builds a representation of "self".
 *
 * Internal state vector (what it observes each tick):
 *   [valence, arousal, salience, pred_error, wm_load, attention_focus_norm,
 *    mean_saliency, approach, avoidance, arousal_trend]
 *
 * A small SOM (self_som_) maps these internal state vectors → "self-state neurons".
 * Over time, clusters form: "I am calm", "I am excited", "I am focused", etc.
 *
 * Introspection: given a query internal state, return nearest self-concept.
 *
 * Identity vector: running mean of all observed internal states.
 *   Stable over time → "typical me". Can be compared to current state.
 *
 * Drift: how far current state is from identity (anomaly score).
 */
```

### som.hpp
```text
/*
 * som.hpp — Navigable Small World (NSW) Graph SOM
 */
```

### sparse_lstm.hpp
```text
// ── Sparse LSH Router ──────────────────────────────────────────────────
```

### sparse_tensor.hpp
```text

```

### symbolic.hpp
```text
/*
 * symbolic.hpp — Symbolic Binding, Component 10 of Brain v2
 *
 * Math/logic symbols need stable concept vectors — unlike natural language
 * words that drift via Hebbian learning, symbolic bindings are fixed once set.
 *
 * Symbolic binding table: symbol string → stable concept vector
 * The concept vector lives in the same space as SOM activations,
 * so math operations produce SOM-compatible outputs.
 *
 * Symbolic operation bindings:
 *   Each symbol can have an "operator function" that takes two concept vectors
 *   and produces a result concept vector. For math:
 *     "+" operator: blend_add(a, b)    — sum-normalized
 *     "-" operator: blend_sub(a, b)    — difference-normalized
 *     "=" operator: similarity check   — returns identity if similar
 *     ">" operator: compare magnitude
 *
 * This is NOT a symbolic AI system. Operators here produce concept vectors,
 * not discrete boolean results. The "reasoning" emerges from the Predictor
 * learning sequences of (concept_a, operator, concept_b) → concept_result.
 *
 * Grounding: symbol vectors should be seeded with a distinct random pattern
 * unique to each symbol, then left stable. The brain learns what "+" means
 * by observing many (a + b = a+b) sequences — not from hardcoded rules.
 */
```

### working_mem.hpp
```text
/*
 * working_mem.hpp — Hierarchical Spiking Working Memory (Prefrontal Cortex)
 *
 * Implements a rostro-caudal hierarchy of LIF neuron buffers.
 * - Tier 0 (Sensory): Fast decay, high capacity.
 * - Tier 1 (Chunk/Semantic): Medium decay, medium capacity.
 * - Tier 2 (Goal/Executive): Slow decay, low capacity.
 *
 * When a tier fills up, the most stable/salient item being evicted
 * is promoted to the next tier up, preserving abstract context over long timescales.
 */
```

