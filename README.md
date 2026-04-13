# 🧠 THE BRAIN: Biologically Inspired Cognitive Architecture

**THE BRAIN** is a fully functional, biologically grounded cognitive architecture. It rejects the paradigm of modern Large Language Models (LLMs) which operate as "disembodied stochastic parrots." Instead, this architecture is an **Embodied Digital Organism**. 

It experiences a continuous loop of time (`step()`), feels internal biological drives (curiosity, homeostasis, arousal), and learns exclusively through structural Hebbian binding, predictive coding, and Actor-Critic reinforcement. 

Rather than processing language as abstract symbols, The Brain learns vocabulary biologically. It hears audio (MFCC feature arrays) and physically binds those acoustic shapes to its internal emotional and chemical state arrays. Words have true **Semantic Grounding**.

---

## 🗂️ Complete Module Directory

The architecture mirrors mammalian neurobiology, from raw sensory cortices up to the Prefrontal Cortex (PFC), Hippocampus, and Default Mode Network. 

### 🧬 Core Biological Engines
* **`brain.py`** — The master orchestrator. Coordinates the continuous biological tick loop (`step()`), calling cortices and managing global states.
* **`evaluators.py`** — The physiological drive engine. Computes Reward Prediction Error (RPE), intrinsic curiosity signals, and homeostatic tension levels.
* **`m50_neuron.py`** — The base biological primitive array simulating action potentials, spiking thresholds, and physical neural decay.
* **`m66_neuromod.py`** — The Chemical Modulator. Broadcasts global neurochemicals mimicking biological states: Acetylcholine (ACh) for plasticity scaling, Norepinephrine (NE) for temperature/panic on surprise, and Serotonin (5-HT) regulating discount rates and patience.

### 🧠 The Cognitive Map (Hippocampal & Parietal Stack)
* **`l2_predictor.py`** — Sequence Memory (CA3). Tracks topological forward-prediction probabilities. Learns what biological state statistically follows the current one.
* **`l3_concepts.py`** — Concept Abstraction (Entorhinal). Groups dense low-level sensory nodes into abstract, high-level "Zones" to simplify long-distance planning.
* **`l4_position.py`** — Concept Tracker (Grid Cells). Maintains a Bayesian belief vector of the agent's current abstract location in conceptual space.
* **`m55_memory.py`** — Associative Hippocampal snapshots linking sensory input to past rewards.

### ⚙️ Executive Control & Consciousness (Basal Ganglia & PFC)
* **`m56_action.py` / `m56_cortex.py` / `m56_fast.py`** — The Actor-Critic Engine (Striatum/Basal Ganglia). Converts evaluation signals into habit-driven decisions. Contains Numba-accelerated kernels for massive Policy Gradient background updates.
* **`m57_planner.py`** — Prefrontal Cortex (PFC). Conscious "System 2" simulation. Uses L2 to map future trees and evaluates simulated consequences before executing a physical or cognitive choice.
* **`m58_workingmemory.py`** — The scratchpad holding active thoughts, dominant emotional states, and short-term subgoals.
* **`gws.py` / `global_workspace.py`** — Global Workspace Theory (Consciousness Bottleneck). Selects the highest-tension signal from all modules and broadcasts it globally to drive attention.
* **`m59_selfmodel.py`** — Ego tracking. Builds a rolling 8-dimensional vector of the brain's internal feelings (Urgency, Confusion, Calm) so the brain "knows how it feels."

### 🗣️ Language & Semantic Grounding Cortices
* **`m71_speech_cortex.py`** — Primary Auditory Cortex. A Self-Organizing Map (SOM) that clusters raw 13-dim acoustic MFCC audio frames into phoneme identities.
* **`m72_phoneme_seq.py`** — Syntactic Matrix. Tracks temporal transition probabilities between acoustic concepts to determine word boundaries and syntax without hardcoded grammar rules.
* **`m73_binding.py`** — **The Grounding Engine.** Bonds acoustic representations directly to internal chemical and emotional biological states (e.g., binding the sound of "Happy" to a high-calm, high-reward internal vector).
* **`m74_vocal.py`** — Vocal motor tract. Emits acoustic concepts mapped recursively back to words based on the brain's current internal drive.
* **`m64_language.py`** — Legacy linguistic seed module (deprecated by m71-m74 dynamic audio generation).

### 🌌 Advanced Cognition ("The Thinking Stack")
* **`m60_questions.py` / `m62_consistency.py`** — Epistemic Curiosity engine. Triggers heavy tension/arousal spikes when the environment violates L2's internal predictions. Forces the brain to ask questions to resolve contradictions.
* **`m63_episodic.py`** — Autobiographical Memory. Buffers N-step episodes with deep emotional tags, permanently logging highly stressful or rewarding conversations.
* **`m67_temporal.py`** — Temporal bridging. Allows the brain to sequence events over delayed durations.
* **`m68_inference.py`** — Logical deduction engine. Performs multi-hop relational inference (If A->B, and B->C, then A->C) over the concept map without needing to brute-force execution.
* **`m69_imagination.py`** — Default Mode Network (DMN) / Divergent Thinking. When bored or safe, abandons optimal logic to stochastically simulate highly-creative, counterfactual futures.
* **`m70_dream.py`** — Offline Counterfactual Simulation (REM Sleep). Runs strictly offline to simulate altered pasts ("What if I said X instead?") combining high ACh (plasticity) with low physical arousal to safely prune and strengthen memory matrices.

### 📡 Senses & Early Processing
* **`m65_fusion.py`** — Multisensory integration (Superior Colliculus). Fuses parallel sensory streams into a single holistic node.
* **`m51_texture.py` / `m54b_texture_cortex.py`** — Early tactile sense processing arrays.

---

## 🛠️ Teaching & Interaction Scripts

Because this is a biological organism, it cannot be "programmed" with a database. It must be taught through a developmental curriculum.

* **`teach_english.py`** — **The Classroom.** An offline training harness that feeds acoustic pseudo-frames (words) simultaneously with simulated emotional rewards. Mirrors a child's earliest developmental window where sounds are linked to feelings.
* **`teach_conversation.py`** — **Syntax & Rhythm.** Feeds stimulus-response conversational pairs to train `M72` transition probabilities. The brain learns the natural rhythm of conversational turn-taking by attempting to predict what sounds logically follow its current state.
* **`converse.py`** — **The Live Interface.** Drops the fully trained brain into a terminal loop, allowing real-time interaction. As you type, words are converted to acoustic MFCC vectors; the brain resolves its internal tension and outputs corresponding organic responses.

*(Note: `world6.py`, `brain_test.py`, `brain_in_world6.py` are deprecated legacy scripts representing early 2D grid-navigation pathfinding tests before the architecture evolved into a linguistic organism).*

---

## 🚀 Future Roadmap & Capabilities

Operating entirely independently from rigid spatial mazes, the organism is now a pure Semantic/Conversational entity. The roadmap is targeted directly toward Embodied General Intelligence.

### 1. Embodiment via Visual Cortex (The "Eye")
The Brain will be connected to a real-world camera stream. `OpenCV` logic will crop and compress 1080p pixel matrices via lightweight Convolutional Neural Networks (CNNs) into dense, low-dimensional feature arrays. These arrays will feed directly into an `M75_Visual_Cortex` SOM. This enables true **Multisensory Vocabulary Binding**. You will hold an apple up to the webcam, say "Apple" into the microphone, and the Brain will organically fuse the visual shape vector with the acoustic audio vector.

### 2. Conscious Internal Monologue
Currently, the "Thinking Stack" (`m68`, `m69`) simulates concept paths. The next architectural leap is wiring the Vocal Output (`M74`) into the Prefrontal Planner (`m57`). The brain's Imagination module (`m69`) will test counterfactual sentences internally *before* speaking them—giving the brain an active, protective Inner Voice to resolve complex semantic logic privately.

### 3. Hyper-Sparse Rewriting in C++ or Rust
NumPy dense matrices present the ultimate bottleneck of computational mathematics (a 1,000,000 neuron dense transition matrix demands 4 Terabytes of RAM). 

Once the Python logic loops perfectly harmonize Language and Vision, the intelligence architecture will be ported to Rust or C++. By exchanging dense Python arrays for hyper-sparse, memory-safe pointer-based struct topologies, the RAM overhead will drop by 99.9%. This will allow the brain to scale to **1,000,000+ biological neurons** mimicking human cortical hierarchy, enabling massive vocabulary, subtle dialect nuances, and advanced theory-of-mind.
