#pragma once
/*
 * brain.hpp — Integration Layer, Brain v2
 *
 * Wires all 10 components into a unified cognitive loop:
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

#include "som.hpp"
#include "predictor.hpp"
#include "episodic.hpp"
#include "working_mem.hpp"
#include "language.hpp"
#include "imagination.hpp"
#include "emotion.hpp"
#include "attention.hpp"
#include "self_model.hpp"
#include "symbolic.hpp"
#include "scratchpad.hpp"
#include "reasoning.hpp"

#include <vector>
#include <string>
#include <memory>
#include <mutex>

namespace brain2 {

struct PerceiveResult {
    int   bmu;                    // SOM best matching unit
    float prediction_error;       // predictor error
    bool  attention_passed;       // did it pass attention gate?
    bool  episodic_stored;        // was an episode committed?
    float valence;                // current emotion valence
    float arousal;                // current emotion arousal
    float salience;               // attention salience
    int   self_concept;           // current self-model concept
};

struct ThinkResult {
    std::vector<std::string>       words;      // words generated this think step
    std::vector<std::vector<float>> concepts;  // concept vectors this step
    float coherence;                           // mean cosine sim of concept sequence
};

class Brain {
public:
    int n_dims;    // SOM feature dimensionality
    int som_rows;
    int som_cols;

    // All 10 components — public for direct access in training loops
    SOM            som;
    Predictor      predictor;
    EpisodicMemory episodic;
    WorkingMemory  working_mem;
    Language       language;
    Imagination    imagination;
    Emotion        emotion;
    Attention      attention;
    SelfModel      self_model;
    Symbolic       symbolic;
    Scratchpad     scratchpad;
    ReasoningEngine reasoning;

private:
    std::unique_ptr<std::mutex> mtx_;
    int                         step_;
    std::vector<float>          prev_act_map_;   // buffered for 1-step-ahead prediction
    bool                        have_prev_act_ = false;
    std::vector<float>          last_act_map_;   // last SOM activation for grounding

    static float cosine(const std::vector<float>& a,
                        const std::vector<float>& b) noexcept {
        float dot = 0.f, na = 0.f, nb = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    InternalState build_internal_state() const {
        InternalState s;
        s.valence         = emotion.valence;
        s.arousal         = emotion.arousal;
        s.salience        = emotion.salience();
        s.pred_error      = predictor.last_error();
        s.wm_load         = working_mem.empty() ? 0.f :
                            float(working_mem.size()) / float(working_mem.capacity);
        // attention focus as normalized index
        float focus = (som.n_neurons > 1)
            ? float(attention.focus_neuron()) / float(som.n_neurons - 1)
            : 0.f;
        s.attention_focus = focus;
        s.mean_saliency   = attention.mean_saliency();
        s.approach        = emotion.approach_mode() ? 1.f : 0.f;
        s.avoidance       = emotion.avoidance_mode() ? 1.f : 0.f;
        s.arousal_trend   = self_model.arousal_trend();
        return s;
    }

public:
    Brain() : n_dims(0), som_rows(0), som_cols(0), step_(0),
              mtx_(std::make_unique<std::mutex>()) {}

    Brain(int som_rows, int som_cols, int n_dims,
          int hidden_dim       = 256,
          int wm_capacity      = 7,
          int episodic_max     = 2000,
          int self_neurons     = 16,
          unsigned seed        = 42)
        : n_dims(n_dims),
          som_rows(som_rows),
          som_cols(som_cols),
          som(som_rows, som_cols, n_dims, 0.15f, 0.9998f, 0.9999f, seed),
          predictor(som_rows * som_cols, hidden_dim, 0.001f, seed),
          episodic(som_rows * som_cols, episodic_max, 0.3f),
          working_mem(som_rows * som_cols, wm_capacity, 0.95f),
          language(som_rows * som_cols, 0.05f),
          imagination(&predictor, 50),
          emotion(0.05f, 0.01f),
          attention(som_rows * som_cols, 0.1f, 0.3f),
          self_model(self_neurons, seed),
          symbolic(som_rows * som_cols),
          scratchpad(som_rows * som_cols),
          reasoning(&symbolic, som_rows * som_cols, 50, 0.01f, &predictor),
          step_(0),
          mtx_(std::make_unique<std::mutex>()) {
        symbolic.seed_math_symbols();
    }

    Brain(Brain&&)            = default;
    Brain& operator=(Brain&&) = default;
    Brain(const Brain&)       = delete;
    Brain& operator=(const Brain&) = delete;

    // PERCEIVE: process one raw input vector through full pipeline
    PerceiveResult perceive(const std::vector<float>& input) {
        int N = som.n_neurons;

        // 1. SOM: find BMU + activation map
        int bmu = som.find_bmu(input);
        auto act_map = som.activation_map(input);
        som.update(input, bmu, 1.f + emotion.lr_modulator() * 0.5f);

        // 2. Predictor: 1-step-ahead prediction
        //    Input = prev act_map, actual = current act_map
        //    First step: no actual (can't know future yet)
        std::vector<float> pred_next;
        if (have_prev_act_) {
            pred_next = predictor.step(prev_act_map_, &act_map);
        } else {
            pred_next = predictor.step(act_map);
        }
        prev_act_map_  = act_map;
        last_act_map_  = act_map;
        have_prev_act_ = true;
        float error = predictor.last_error();

        // 3. Emotion: update from surprise
        emotion.from_prediction_error(error);
        emotion.tick();

        // 4. Attention: gate by novelty + emotional arousal
        auto attn_result = attention.gate(act_map, error,
                                          emotion.attention_modulator());

        // 5. WorkingMemory: insert if attention passed
        if (attn_result.passed) {
            working_mem.gate(act_map, emotion.salience());
        }
        working_mem.tick();

        // 6. EpisodicMemory: observe all activations, commit if surprising
        episodic.observe(act_map);
        bool stored = episodic.commit(error);

        // 7. SelfModel: observe internal state
        auto istate = build_internal_state();
        self_model.observe(istate);

        step_++;

        PerceiveResult r;
        r.bmu              = bmu;
        r.prediction_error = error;
        r.attention_passed = attn_result.passed;
        r.episodic_stored  = stored;
        r.valence          = emotion.valence;
        r.arousal          = emotion.arousal;
        r.salience         = attn_result.score;
        r.self_concept     = self_model.current_concept(istate);
        return r;
    }

    // Reset sequence boundary (call between unrelated sequences)
    void reset_sequence() {
        predictor.reset();
        have_prev_act_ = false;
    }

    // HEAR: hear a word grounded to last SOM activation (not blended WM context)
    // This ensures word vectors learn clean concept activations, not blends.
    void hear(const std::string& word) {
        if (!last_act_map_.empty()) {
            language.hear(word, last_act_map_);
        } else {
            auto ctx = working_mem.context();
            if (ctx.empty()) ctx = std::vector<float>(som.n_neurons, 0.f);
            language.hear(word, ctx);
        }
    }

    // THINK: run N inner speech steps
    // Each step: WM context → decode → best word → re-encode → predict → push to WM
    ThinkResult think(int steps = 5) {
        ThinkResult result;
        if (working_mem.empty()) return result;

        auto ctx = working_mem.context();
        result.concepts.push_back(ctx);

        predictor.reset();
        predictor.set_offline(true);

        float total_coh = 0.f;
        int   coh_count = 0;

        for (int i = 0; i < steps; i++) {
            // Decode current context to a word
            auto word = language.best_word(ctx);
            if (!word.empty()) {
                result.words.push_back(word);
                // Re-encode to get concept vector
                auto word_vec = language.encode(word);
                // Blend word concept back into working memory
                working_mem.gate(word_vec, 0.f);
                working_mem.tick();
                ctx = working_mem.context();
            }

            // Predict next activation
            auto next_ctx = predictor.step(ctx);
            // Coherence between steps
            float coh = cosine(ctx, next_ctx);
            total_coh += coh;
            coh_count++;

            ctx = next_ctx;
            result.concepts.push_back(ctx);
        }

        predictor.set_offline(false);
        result.coherence = (coh_count > 0) ? total_coh / float(coh_count) : 0.f;
        return result;
    }

    // SPEAK: convert concept sequence to word sequence
    std::vector<std::string> speak(const std::vector<std::vector<float>>& concepts,
                                   float min_sim = 0.f) {
        return language.speak(concepts, min_sim);
    }

    // DREAM: rest-phase consolidation
    // Returns high-coherence dream frames for optional inspection
    std::vector<std::vector<float>> dream(int n_dreams = 20,
                                          int steps_per_dream = 15) {
        // Get seed memories from episodic
        std::vector<std::vector<float>> seeds;
        auto ctx = working_mem.context();
        if (!working_mem.empty()) {
            // Retrieve top memories similar to current context
            auto topk = episodic.retrieve_topk(ctx, 5);
            for (auto& [sim, idx] : topk) {
                auto* ep = episodic.retrieve(ctx);
                if (ep && !ep->frames.empty())
                    seeds.push_back(ep->frames[0]);
            }
        }

        // Run dreams
        auto dreams = imagination.dream(n_dreams, steps_per_dream, seeds, 42u);

        // Extract high-coherence frames → working memory
        auto frames = imagination.extract_frames(dreams, 0.6f);
        for (const auto& f : frames)
            working_mem.gate(f, 0.f);

        // Consolidate episodic memory
        episodic.consolidate(0.85f);

        return frames;
    }

    // Evaluate a goal state against imagination
    float imagine_goal(const std::vector<float>& start,
                       const std::vector<float>& goal,
                       int steps = 20) {
        auto sim = imagination.simulate(start, steps);
        return imagination.evaluate(sim, goal, 0.8f);
    }

    // Lookup symbolic binding
    std::vector<float> symbol(const std::string& sym) {
        return symbolic.lookup(sym);
    }

    // Apply symbolic operation
    std::vector<float> symbolic_op(const std::string& op_sym,
                                    const std::vector<float>& a,
                                    const std::vector<float>& b) {
        return symbolic.apply(op_sym, a, b);
    }

    int  step()        const noexcept { return step_; }
    bool initialized() const noexcept { return n_dims > 0; }
};

} // namespace brain2
