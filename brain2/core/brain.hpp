#pragma once
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
// Brain V3 additions
#include "predictive_coding.hpp"
#include "binding_memory.hpp"
#include "global_workspace.hpp"
#include "basal_ganglia.hpp"
#include "procedural_memory.hpp"
#include "hierarchical_predictor.hpp"
#include "analogy.hpp"

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <algorithm>

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

    // All 16 components — public for direct access in training loops
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
    // Brain V3 & V4 additions
    PredictiveCodingLayer pc_som;        // wraps SOM output
    PredictiveCodingLayer pc_hpred;      // wraps H-Predictor output
    PredictiveCodingLayer pc_wm;         // wraps Working Memory output
    PredictiveCodingLayer pc_bg;         // wraps BG controller input
    BindingMemory         binding;       // hippocampal triple storage
    AnalogyEngine         analogy;       // structure mapping analogy
    GlobalWorkspace       global_ws;     // lateral inhibition
    BasalGanglia          bg_controller; // learned op selector
    ProceduralMemory      procedures;    // reusable strategies
    HierarchicalPredictor h_predictor;  // chunk + episode predictors

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
                            float(working_mem.size()) / float(working_mem.get_base_capacity() * 1.5f + 1e-5f);
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
          predictor(som_rows * som_cols, hidden_dim, 0.005f, seed),
          episodic(som_rows * som_cols, episodic_max, 0.3f),
          working_mem(som_rows * som_cols, wm_capacity, 0.95f),
          language(n_dims, 0.05f),
          imagination(&predictor, 50),
          emotion(0.05f, 0.01f),
          attention(som_rows * som_cols, 0.1f, 0.3f),
          self_model(self_neurons, seed),
          symbolic(som_rows * som_cols),
          scratchpad(som_rows * som_cols),
          reasoning(&symbolic, som_rows * som_cols, 50, 0.01f, &predictor),
          // V3 & V4 components
          pc_som(som_rows * som_cols, 0.05f, 0.01f),
          pc_hpred(som_rows * som_cols, 0.05f, 0.01f),
          pc_wm(som_rows * som_cols, 0.05f, 0.01f),
          pc_bg(som_rows * som_cols, 0.05f, 0.01f),
          binding(som_rows * som_cols, 2000),
          analogy(&binding),
          global_ws(som_rows * som_cols),
          bg_controller(som_rows * som_cols, 0.001f, seed),
          procedures(som_rows * som_cols),
          h_predictor(som_rows * som_cols, 128, 64, seed),
          step_(0),
          mtx_(std::make_unique<std::mutex>()) {
        symbolic.seed_math_symbols();
    }

    Brain(Brain&&)            = default;
    Brain& operator=(Brain&&) = default;
    Brain(const Brain&)       = delete;
    Brain& operator=(const Brain&) = delete;

    void load_components(const std::string& predictor_path,
                         const std::string& language_path,
                         const std::string& som_path,
                         const std::string& episodic_path,
                         const std::string& emotion_path,
                         const std::string& self_path,
                         const std::string& symbolic_path,
                         // V3 components (optional — empty string = skip)
                         const std::string& binding_path    = "",
                         const std::string& bg_path         = "",
                         const std::string& procedures_path = "",
                         const std::string& hpred_path      = "") {
        predictor  = Predictor::load(predictor_path);
        language   = Language::load(language_path);
        som        = SOM::load(som_path);
        episodic   = EpisodicMemory::load(episodic_path);
        emotion    = Emotion::load(emotion_path);
        self_model = SelfModel::load(self_path);
        symbolic   = Symbolic::load(symbolic_path);

        n_dims      = som.n_dims;
        som_rows    = som.rows;
        som_cols    = som.cols;
        imagination = Imagination(&predictor, 50);
        reasoning   = ReasoningEngine(&symbolic, som.n_neurons, 50, 0.01f, &predictor);
        prev_act_map_.clear();
        last_act_map_.clear();
        have_prev_act_ = false;

        // V3 optional loads
        if (!binding_path.empty())    binding      = BindingMemory::load(binding_path);
        if (!bg_path.empty())         bg_controller = BasalGanglia::load(bg_path);
        if (!procedures_path.empty()) procedures   = ProceduralMemory::load(procedures_path);
        if (!hpred_path.empty())      h_predictor  = HierarchicalPredictor::load(hpred_path);
    }

    // PERCEIVE: process one raw input vector through full pipeline (Brain V4)
    PerceiveResult perceive(const std::vector<float>& input) {
        // 1. SOM: find BMU + activation map
        int bmu      = som.find_bmu(input);
        auto act_map = som.activation_map(input);
        som.update(input, bmu, 1.f + emotion.lr_modulator() * 0.5f);

        // 2. Predictive coding 1: SOM output -> Error signal
        auto pc1_err = pc_som.propagate(act_map);
        bool do_propagate = pc_som.should_propagate();
        pc_som.update();

        // 3. Predictor (Fast)
        std::vector<float> pred_next;
        if (have_prev_act_) {
            pred_next = predictor.step(prev_act_map_, &pc1_err);
            if (h_predictor.has_prev_chunk_) {
                for (int i = 0; i < n_dims; i++) {
                    pred_next[i] = pred_next[i] * 0.7f + h_predictor.current_chunk_pred_[i] * 0.3f;
                }
            }
        } else {
            pred_next = predictor.step(pc1_err);
        }
        prev_act_map_  = pc1_err;
        last_act_map_  = act_map; // Grounding word vectors needs full act map
        have_prev_act_ = true;
        float error = predictor.last_error();

        // 4. Hierarchical predictor (Processes PC1 Error)
        if (do_propagate) {
            h_predictor.observe(pc1_err);
        }
        
        // 5. Predictive coding 2: H-Predictor -> WM
        auto pc2_err = pc_hpred.propagate(pc1_err);
        pc_hpred.update();
        bool propagate_wm = pc_hpred.should_propagate();

        // 6. Global Workspace competition
        global_ws.bid((int)GWModule::SOM,    pc_som.error_norm, pc2_err);
        global_ws.bid((int)GWModule::PREDICT, 1.f - error,      pred_next);
        global_ws.bid((int)GWModule::EMOTION, emotion.salience(), act_map);
        int gw_winner = global_ws.compete();
        scratchpad.write("attention", global_ws.broadcast(), "gw");

        // 7. Emotion
        emotion.from_prediction_error(error);
        emotion.tick();

        // 8. Attention & Working Memory (Process PC2 Error)
        auto attn_result = attention.gate(pc2_err, error, emotion.attention_modulator());
        bool wm_gate_open = attn_result.passed &&
            (global_ws.is_winner((int)GWModule::SOM) || propagate_wm || error > 0.15f);
        if (wm_gate_open) {
            working_mem.gate(pc2_err, emotion.salience());
        }
        working_mem.tick();

        // 9. Predictive coding 3: WM -> Episodic/Binding
        auto ctx_summary = working_mem.context();
        if (ctx_summary.empty()) ctx_summary.assign(som.n_neurons, 0.f);
        auto pc3_err = pc_wm.propagate(ctx_summary);
        pc_wm.update();

        // 10. EpisodicMemory (Process PC3 Error)
        bool episodic_active = global_ws.is_winner((int)GWModule::SOM) ||
                               global_ws.is_winner((int)GWModule::EMOTION);
        if (episodic_active && pc_wm.should_propagate()) {
            episodic.observe(pc3_err);
        }
        bool stored = episodic_active && episodic.commit(error);

        // 11. Predictive coding 4: WM Context -> BG Controller
        auto pc4_err = pc_bg.propagate(ctx_summary);
        pc_bg.update();

        // 12. Basal Ganglia (Process PC4 Error)
        auto goal_vec = scratchpad.has("goal")
                      ? scratchpad.read("goal")
                      : std::vector<float>(som.n_neurons, 0.f);
        
        Op selected_op = Op::HALT;
        if (pc_bg.should_propagate()) {
            auto* proc = procedures.retrieve(pc4_err);
            if (proc) {
                selected_op = proc->steps.empty() ? Op::HALT : proc->steps[0];
            } else {
                auto bg_act = bg_controller.select_op(pc4_err, goal_vec, /*greedy=*/false);
                selected_op = bg_act.op;
            }
        }
        scratchpad.write("last_op", std::vector<float>{(float)(int)selected_op}, "bg");

        // 13. SelfModel
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

    // Explicit triple binding — call from training loop with known (subject, relation, object)
    // e.g. after processing "dog isa animal": bind(dog_vec, isa_vec, animal_vec)
    void bind_triple(const std::vector<float>& subj,
                     const std::vector<float>& rel,
                     const std::vector<float>& obj) {
        binding.bind(subj, rel, obj);
        scratchpad.write("subject",  subj, "bind");
        scratchpad.write("relation", rel,  "bind");
        scratchpad.write("object",   obj,  "bind");
    }

    // Query BindingMemory by (subject, relation) → object
    // Returns pair (vector, confidence)
    std::pair<std::vector<float>, float> binding_query(const std::vector<float>& a,
                                                       const std::vector<float>& b,
                                                       bool want_object = true,
                                                       float threshold = 0.5f) {
        return binding.query(a, b, want_object, threshold);
    }

    // Perform analogy
    std::vector<float> analogy_op(const std::vector<float>& a,
                                  const std::vector<float>& b) {
        auto ctx = working_mem.context();
        if (ctx.empty()) ctx = std::vector<float>(som.n_neurons, 0.f);
        return analogy.structure_map(a, b, ctx);
    }

    // Reinforce the last BG op-chain (+1 correct, -1 wrong)
    void reinforce_bg(float reward) {
        bg_controller.reinforce(reward);
    }

    // Consolidate current scratchpad op-chain into procedural memory
    void consolidate_procedure(const std::vector<int>& op_ints,
                               const std::string& name = "") {
        std::vector<Op> ops;
        for (int i : op_ints) ops.push_back((Op)i);
        auto ctx = working_mem.context();
        if (!ctx.empty()) procedures.consolidate(ops, ctx, name);
    }

    // Reset sequence boundary (call between unrelated sequences)
    void reset_sequence() {
        predictor.reset();
        working_mem.clear();
        scratchpad.clear();
        have_prev_act_ = false;
        h_predictor.reset();
        bg_controller.clear_traces();
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
        auto current_act = last_act_map_;
        if (current_act.empty()) {
            current_act = working_mem.context();
        }
        if (current_act.empty()) return result;

        result.concepts.push_back(current_act);

        // DO NOT reset predictor here. We want to continue the train of thought from the perceived prompt!
        predictor.set_offline(true);

        float total_coh = 0.f;
        int   coh_count = 0;

        for (int i = 0; i < steps; i++) {
            // Decode current activation to a word
            auto word = language.best_word(current_act);
            if (!word.empty()) {
                result.words.push_back(word);
                // Inner speech feeds back into working memory
                auto word_vec = language.encode(word);
                working_mem.gate(word_vec, 0.f);
                working_mem.tick();
            }

            // Predict next activation
            auto next_act = predictor.step(current_act);
            // Coherence between steps
            float coh = cosine(current_act, next_act);
            total_coh += coh;
            coh_count++;

            current_act = next_act;
            result.concepts.push_back(current_act);
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
                auto* ep = episodic.get_episode(idx);
                if (ep && !ep->root.summary_spike.empty()) {
                    std::vector<float> f_float(ep->root.summary_spike.size(), 0.0f);
                    for (size_t i = 0; i < ep->root.summary_spike.size(); ++i) {
                        if (ep->root.summary_spike[i]) f_float[i] = 1.0f;
                    }
                    seeds.push_back(f_float);
                }
            }
        }

        // Run dreams
        auto dreams = imagination.dream(n_dreams, steps_per_dream, seeds, 42u);

        // Extract high-coherence frames → working memory
        auto frames = imagination.extract_frames(dreams, 0.6f);
        for (const auto& f : frames)
            working_mem.gate(f, 0.f);

        // Sparse Episodic Replay: offline consolidation
        // Train predictor and binding memory using episodic seeds (replay during sleep)
        predictor.set_offline(false);
        for (const auto& seed : seeds) {
            predictor.step(seed);
            if (!ctx.empty()) {
                // Bind surprising events to current context (semantic transfer)
                binding.bind(ctx, ctx, seed);
            }
        }

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
