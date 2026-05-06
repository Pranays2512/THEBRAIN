# Brain v2: Cognitive Architecture & Data Flow

This document provides a detailed technical breakdown of the **Brain v2** system, explaining how data flows through the 10 core components and how the C++ engine interacts with the Python training environment.

---

## 1. System Overview
Brain v2 is a **Predictive Coding** architecture. Unlike traditional LLMs, it separates **sensory grounding** (SOM) from **sequential reasoning** (Predictor) and **symbolic communication** (Language).

### The 10 Core Components
1.  **SOM (Self-Organizing Map):** Maps high-dimensional input vectors into a 2D topological grid (the "Neuron Map").
2.  **Predictor (LSTM):** Learns to predict the next state of the SOM based on current context.
3.  **Language:** Maps SOM activation patterns to human words (Symbols).
4.  **Episodic Memory:** Stores sequences of events that caused high prediction error (Surprise).
5.  **Working Memory:** A salience-gated buffer for "currently active" thoughts.
6.  **Emotion:** Modulates learning rates and attention based on valence/arousal.
7.  **Attention:** Filters inputs so only significant patterns reach Working Memory.
8.  **Self-Model:** Observes the brain's own internal state (metacognition).
9.  **Symbolic/Scratchpad:** Handles discrete operations (Math, Logic).
10. **Imagination:** Uses the Predictor to simulate future outcomes without external input.

---

## 2. The Cognitive Loop (Data Flow)

The brain operates in a sequence of operations called the **Perceive-Think-Speak** cycle.

### Phase 1: PERCEIVE (Input → Internal State)
**Triggered by:** `Brain::perceive(vector)`
**Variable Flow:**
1.  **Raw Vector (`input`)** → `SOM`:
    *   Finds the **BMU** (Best Matching Unit).
    *   Generates an **Activation Map** (`act_map`).
    *   Updates SOM weights toward the input.
2.  **`act_map`** → `Predictor`:
    *   Compares `act_map` to the prediction made in the previous step.
    *   Calculates **`prediction_error`** (Surprise).
    *   Updates LSTM weights to minimize future error.
3.  **`prediction_error`** → `Emotion`:
    *   Increases **Arousal** if error is high.
    *   Updates **Valence** (positive/negative impact).
4.  **`act_map` + `Arousal`** → `Attention`:
    *   If `prediction_error > threshold`, the pattern is deemed "salient."
5.  **Salient `act_map`** → `WorkingMemory`:
    *   The pattern is "gated" into a slot in Working Memory.
    *   Old/weak patterns decay over time.

### Phase 2: THINK (Internal Processing)
**Triggered by:** `Brain::think(steps)`
**Variable Flow:**
1.  **`WM_Context`** (Mean of Working Memory slots) → `Language`:
    *   Decodes the current activation into the **`best_word`**.
2.  **`best_word`** → `Predictor`:
    *   Re-encodes the word vector.
    *   Predicts the **NEXT** likely activation map (Imagination).
3.  **Predicted Map** → `WorkingMemory`:
    *   The brain "talks to itself" by pushing its own predictions back into its Working Memory.

### Phase 3: SPEAK (Output)
**Triggered by:** `Brain::speak(sequence)`
**Variable Flow:**
1.  A sequence of concept vectors is passed to `Language::speak`.
2.  Uses **Cosine Similarity** to find the closest word vectors in the vocabulary.
3.  Returns a string of words.

---

## 3. File-to-File Interaction Mapping

| File | Role | Key Variables Flowing OUT |
| :--- | :--- | :--- |
| **`train.py`** | The Driver | Input data vectors, Learning Rate schedules. |
| **`brain2.cpp`** | The Bridge | Python objects ↔ C++ pointers. |
| **`brain.hpp`** | The Orchestrator | `PerceiveResult`, `ThinkResult`, `act_map`. |
| **`som.hpp`** | The Grounding | BMU indices, Weight matrices. |
| **`predictor.hpp`**| The Logic | `pred_error`, Predicted activation maps. |
| **`language.hpp`** | The Symbolism | Word vectors, Word strings. |

---

## 4. Rationale: Why this order?

1.  **SOM before Predictor:** We must turn high-dimensional "noise" into a structured, low-dimensional "map" before we can learn to predict its movements.
2.  **Predictor before Emotion:** Emotion is fundamentally a reaction to the *validity of our expectations*. We need the prediction error first to know how to feel.
3.  **Attention before Working Memory:** The brain has limited capacity. We filter data through Attention/Emotion first to ensure we only "remember" (WM) what matters.
4.  **Grounded Language:** Notice that `Language` is trained on SOM activations. This means when the brain says "Apple," it isn't just a token—it is linked to a specific region of its sensory "visual" map.

---

## 5. Typical Variable Values (Sanity Check)

*   **`n_dims`:** Usually **64 to 256**. (Feature complexity).
*   **`som_rows/cols`:** **32 to 64**. (Mental map resolution).
*   **`prediction_error`:** Starts near **1.0**, targets **< 0.1**.
*   **`valence/arousal`:** Range **[-1.0, 1.0]**.
*   **`cosine_similarity`:** **> 0.8** for good word matches.
