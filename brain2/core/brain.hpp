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
#include "decoder.hpp"
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
#include <numeric>
#include <iostream>
#include <memory>
#include <mutex>
#include <algorithm>
#include <chrono>

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
    DecoderRNN            decoder;       // generative sequence decoder
    std::vector<std::string> spoken_words;

    std::vector<std::string> get_spoken_words() {
        return spoken_words;
    }
    
    void clear_spoken_words() {
        spoken_words.clear();
    }

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
                         const std::string& hpred_path      = "",
                         const std::string& decoder_path    = "") {
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
        if (!decoder_path.empty())    decoder      = DecoderRNN::load(decoder_path);
    }

    void save_components(const std::string& directory) const {
        predictor.save(directory + "/predictor.bin");
        language.save(directory + "/language.bin");
        som.save(directory + "/som.bin");
        episodic.save(directory + "/episodic.bin");
        emotion.save(directory + "/emotion.bin");
        self_model.save(directory + "/self.bin");
        symbolic.save(directory + "/symbolic.bin");
        
        // V3 Optional Components (save if active)
        binding.save(directory + "/binding.bin");
        bg_controller.save(directory + "/bg.bin");
        procedures.save(directory + "/procedures.bin");
        h_predictor.save(directory + "/hpred.bin");
        decoder.save(directory + "/decoder.bin");
    }

    bool commit_episode(float err, const std::vector<float>& payload = {}) {
        return episodic.commit(err, payload);
    }

    // PERCEIVE: process one raw input vector through full pipeline
    PerceiveResult perceive(const std::vector<float>& input) {
        std::lock_guard<std::mutex> lock(*mtx_);
        scratchpad.write("sensory_input", input, "sensory");

        // 1. SOM: find BMU + activation map
        int bmu      = som.find_bmu(input);
        auto act_map = som.activation_map(input);
        som.update(input, bmu, 1.f + emotion.lr_modulator() * 0.5f);
        
        // Add to Episodic Memory
        episodic.observe(act_map);

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

    // ── Unsupervised Daydreaming (Phase 1) ──────────────────────────────────
    // Runs the predictive coding cycle on internally generated "imagination"
    // Updates weights unsupervised to consolidate models without real input.
    void daydream() {
        std::lock_guard<std::mutex> lock(*mtx_);
        
        // 1. Generate a short dream sequence (5 steps) from random noise in concept space (64-dim)
        std::vector<std::vector<float>> empty_seeds;
        unsigned int seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
        auto sims = imagination.dream(1, 5, empty_seeds, seed);
        auto frames = imagination.extract_frames(sims, 0.0f); 
        
        if (frames.size() < 2) return;
        
        bool was_offline = predictor.is_offline();
        predictor.set_offline(false);
        
        // frames[0] is already an activation map (output of predictor or noise of same size)
        auto act_map_0 = frames[0];
        std::vector<float> prev_err = pc_som.propagate(act_map_0);
        pc_som.update();
        
        // 2. Step through the dream sequence
        for (size_t i = 1; i < frames.size(); i++) {
            auto act_map = frames[i];
            auto curr_err = pc_som.propagate(act_map);
            pc_som.update();
            
            // Train predictor: input is prev_err, target is curr_err
            predictor.step(prev_err, &curr_err);
            
            prev_err = curr_err;
            
            // Emotion responds to internal prediction errors
            float error = predictor.last_error();
            emotion.from_prediction_error(error);
            emotion.tick();
        }
        
        predictor.set_offline(was_offline);
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

    // Consolidate current scratchpad op-chain into procedural memory.
    // Uses the GOAL WORD embedding as the trigger (not SOM context) — this is
    // stable across SOM updates and makes retrieval deterministic by intent.
    void consolidate_procedure(const std::vector<int>& op_ints,
                               const std::string& name = "") {
        std::vector<Op> ops;
        for (int i : op_ints) ops.push_back((Op)i);
        // Prefer goal-word vector as trigger; fall back to WM context if name unknown
        std::vector<float> trigger;
        if (!name.empty() && language.knows(name)) {
            trigger = language.encode(name);
        } else {
            trigger = working_mem.context();
        }
        if (!trigger.empty()) procedures.consolidate(ops, trigger, name);
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

    std::vector<float> simulate_op(Op op, Scratchpad& pad, bool commit = true) {
        switch (op) {
            case Op::READ: {
                auto res = pad.read("result");
                pad.write("focus", res, "attn");
                break;
            }
            case Op::WRITE: {
                auto focus = pad.read("focus");
                if (!focus.empty()) pad.write("subject", focus, "mem");
                break;
            }
            case Op::MATH_SUB: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::stoi(s_sym) - std::stoi(o_sym);
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("result", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_DIV: {
                auto res = pad.read("result");
                auto rel = pad.read("relation");
                if (!res.empty() && !rel.empty()) {
                    auto r_sym = language.best_word(res);
                    auto d_sym = language.best_word(rel);
                    if (!r_sym.empty() && !d_sym.empty()) {
                        try {
                            int d_val = std::stoi(d_sym);
                            if (d_val != 0) {
                                int fin = std::stoi(r_sym) / d_val;
                                std::string fin_sym = std::to_string(fin);
                                if (!language.knows(fin_sym)) {
                                    language.register_word(fin_sym);
                                    symbolic.bind(fin_sym);
                                }
                                pad.write("result", language.encode(fin_sym), "math");
                            }
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::STORE_TMP: {
                auto res = pad.read("result");
                if (!res.empty()) {
                    pad.write("relation", res, "math");
                }
                break;
            }
            case Op::MATH_ADD: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::stoi(s_sym) + std::stoi(o_sym);
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("result", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_MUL: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::stoi(s_sym) * std::stoi(o_sym);
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("result", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_POW: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::pow(std::stoi(s_sym), std::stoi(o_sym));
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("result", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_FACT: {
                auto subj = pad.read("subject");
                if (!subj.empty()) {
                    auto s_sym = language.best_word(subj);
                    if (!s_sym.empty()) {
                        try {
                            int n = std::stoi(s_sym);
                            long long res = 1;
                            for (int i = 2; i <= n; i++) res *= i;
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("result", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_FACT_REL: {
                auto rel = pad.read("relation");
                if (!rel.empty()) {
                    auto r_sym = language.best_word(rel);
                    if (!r_sym.empty()) {
                        try {
                            int n = std::stoi(r_sym);
                            long long res = 1;
                            for (int i = 2; i <= n; i++) res *= i;
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("relation", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_DIV_FLOAT: {
                // Float division: subject / object → written as 2-decimal string e.g. "0.17"
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            float denom = std::stof(o_sym);
                            if (std::abs(denom) > 1e-6f) {
                                float val = std::stof(s_sym) / denom;
                                // Format to 2 decimal places
                                char buf[32];
                                std::snprintf(buf, sizeof(buf), "%.2f", val);
                                std::string res_sym(buf);
                                if (!language.knows(res_sym)) {
                                    language.register_word(res_sym);
                                    symbolic.bind(res_sym);
                                }
                                pad.write("result", language.encode(res_sym), "math");
                            }
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::COMPARE: {
                auto result = pad.read("result");
                auto obj    = pad.read("object"); // Compare result with queried object
                float sim = 0.f;
                if (!result.empty() && !obj.empty()) {
                    sim = cosine(result, obj);
                }
                std::vector<float> sim_vec(n_dims, sim);
                pad.write("comparison", sim_vec, "eval");
                break;
            }
            case Op::NOT: {
                auto obj = pad.read("object");
                if (!obj.empty()) {
                    for (auto& x : obj) x = -x;
                    pad.write("object", obj, "modifier");
                }
                break;
            }
            case Op::BIND_QUERY: {
                auto subj = pad.read("subject");
                auto rel  = pad.read("relation");
                if (!subj.empty() && !rel.empty()) {
                    auto [ans, conf] = binding.query(subj, rel, true);
                    if (conf > 0.5f) {
                        pad.write("result", ans, "binding");
                    }
                }
                break;
            }
            case Op::BIND_ISA: {
                auto subj = pad.read("subject");
                if (!subj.empty()) {
                    auto isa_vec = language.encode("isa");
                    auto [ans, conf] = binding.query(subj, isa_vec, true);
                    if (conf > 0.5f) {
                        pad.write("result", ans, "binding");
                    }
                }
                break;
            }
            case Op::RETRIEVE: {
                auto focus = pad.read("focus");
                if (!focus.empty()) {
                    auto topk = episodic.retrieve_topk(focus, 1);
                    if (!topk.empty()) {
                        auto* ep = episodic.get_episode(topk[0].second);
                        if (ep) {
                            if (!ep->payload.empty()) {
                                pad.write("result", ep->payload, "episodic");
                            } else if (!ep->root.summary_spike.empty()) {
                                std::vector<float> dense(n_dims, 0.f);
                                size_t lim = std::min(dense.size(), ep->root.summary_spike.size());
                                for (size_t i = 0; i < lim; i++) {
                                    if (ep->root.summary_spike[i]) dense[i] = 1.0f;
                                }
                                pad.write("result", dense, "episodic");
                            }
                        }
                    }
                }
                break;
            }
            case Op::ANALOGY: {
                auto a   = pad.read("subject");
                auto rel = pad.read("relation");
                auto ctx = working_mem.context();
                if (!a.empty() && !rel.empty()) {
                    auto mapped = analogy.structure_map(a, rel, ctx.empty() ? std::vector<float>(n_dims, 0.f) : ctx);
                    pad.write("result", mapped, "analogy");
                }
                break;
            }
            case Op::ASK_USER: {
                // Signals that the BG wants to ask the user a question
                // Writes a special "ask" goal to result
                auto ask_vec = language.encode("ask");
                pad.write("result", ask_vec, "curiosity");
                break;
            }
            case Op::STORE_SUBJ: {
                auto sens = pad.read("sensory_input");
                if (!sens.empty()) pad.write("subject", sens, "context");
                break;
            }
            case Op::STORE_REL: {
                auto sens = pad.read("sensory_input");
                if (!sens.empty()) pad.write("relation", sens, "context");
                break;
            }
            case Op::STORE_OBJ: {
                auto sens = pad.read("sensory_input");
                if (!sens.empty()) pad.write("object", sens, "context");
                break;
            }
            case Op::SPEAK: {
                auto res = pad.read("result");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_SUBJ: {
                auto res = pad.read("subject");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_REL: {
                auto res = pad.read("relation");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_OBJ: {
                auto res = pad.read("object");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::ATTEND: {
                // Focus on the missing slot that contains '?'
                auto subj = pad.read("subject");
                auto rel = pad.read("relation");
                auto obj = pad.read("object");
                
                auto q_vec = language.encode("?");
                
                if (!subj.empty() && cosine(subj, q_vec) > 0.8f) {
                    pad.write("focus", subj, "attention");
                } else if (!rel.empty() && cosine(rel, q_vec) > 0.8f) {
                    pad.write("focus", rel, "attention");
                } else if (!obj.empty() && cosine(obj, q_vec) > 0.8f) {
                    pad.write("focus", obj, "attention");
                }
                break;
            }
            case Op::HALT:
            default:
                break;
        }

        std::vector<float> ctx(n_dims, 0.f);
        auto slots = pad.slot_names();
        for (const auto& s : slots) {
            auto val = pad.read(s);
            
            // Create a deterministic slot-specific embedding
            std::string slot_sym = "SLOT_" + s;
            if (!symbolic.knows(slot_sym)) symbolic.bind(slot_sym);
            auto slot_embed = symbolic.lookup(slot_sym);
            
            size_t lim = std::min(val.size(), ctx.size());
            for (size_t i = 0; i < lim; i++) {
                ctx[i] += val[i] * slot_embed[i];
            }
        }
        float norm_val = 0.f;
        for (float x : ctx) norm_val += x * x;
        if (norm_val > 1e-8f) {
            norm_val = std::sqrt(norm_val);
            for (float& x : ctx) x /= norm_val;
        }
        return ctx;
    }

    void load_bg(const std::string& path) {
        bg_controller = BasalGanglia::load(path);
    }

    void save_bg(const std::string& path) const {
        bg_controller.save(path);
    }

    void start_reasoning() {
        bg_controller.clear_traces();
        std::vector<float> initial_ctx = simulate_op(Op::HALT, scratchpad);
        scratchpad.start_tree(initial_ctx);
    }

    std::vector<int> reason(const std::string& goal_word, int max_steps = 10, float epsilon = 0.0f) {
        std::vector<int> solution_path;
        auto goal_vec = language.encode(goal_word);
        
        std::vector<float> initial_ctx = simulate_op(Op::HALT, scratchpad); // just gets current summary
        int root_id = scratchpad.start_tree(initial_ctx);
        
        for (int step = 0; step < max_steps; step++) {
            auto current_ctx = scratchpad.current_tree_state();
            
            bool greedy = (epsilon == 0.0f);
            auto act = bg_controller.select_op(current_ctx, goal_vec, greedy, -1);
            
            // Re-run forward pass to get value just for logging/halting
            std::vector<float> h, inp;
            auto [logits, current_value] = bg_controller.forward(current_ctx, goal_vec, h, inp);
            
            if (current_value >= 0.95f) {
                solution_path.push_back((int)Op::HALT);
                break; // Reached goal!
            }
            
            struct Eval { int op; float cost; int child_id; };
            std::vector<Eval> evals;
            auto probs = BasalGanglia::softmax(logits);
            
            for (int op_idx = 0; op_idx < (int)Op::N_OPS; op_idx++) {
                if (op_idx == (int)Op::HALT) continue;
                if (probs[op_idx] < 0.05f) continue;
                
                auto sim_pad = scratchpad; // Deep copy
                auto next_ctx = simulate_op((Op)op_idx, sim_pad, false);
                
                std::vector<float> h_child, inp_child;
                auto [logits_c, value_c] = bg_controller.forward(next_ctx, goal_vec, h_child, inp_child);
                
                // PUCT style heuristic: combine Critic value with Actor prior
                float c_puct = 2.0f;
                float h_cost = (1.0f - value_c) - c_puct * probs[op_idx];
                
                int child_id = scratchpad.branch(next_ctx, h_cost);
                evals.push_back({op_idx, h_cost, child_id});
            }
            
            if (evals.empty()) break;
            
            int best_id = scratchpad.move_to_best_child();
            int picked_op = (int)Op::HALT;
            for (auto& e : evals) {
                if (e.child_id == best_id) {
                    picked_op = e.op;
                    break;
                }
            }
            
            solution_path.push_back(picked_op);
            simulate_op((Op)picked_op, scratchpad);
        }
        return solution_path;
    }

    // Unified Cognitive Step: PERCEIVE -> THINK -> SPEAK
    std::string cognitive_step(const std::string& input_text) {
        // 1. PERCEIVE
        auto input_vec = language.encode(input_text);
        auto perceive_res = perceive(input_vec);
        
        // 2. THINK (Reasoning)
        // Load working memory context into scratchpad to ground the tree search
        auto ctx = working_mem.context();
        if (!ctx.empty()) {
            scratchpad.write("subject", ctx, "context");
        }
        
        // We set a default conversational goal: resolve uncertainty or reply
        std::string goal_word = "reply";
        
        // Use PUCT to reason for up to 5 steps
        std::vector<int> solution = reason(goal_word, 5, 0.05f); // 5% epsilon exploration
        
        // 3. SPEAK
        // After reasoning, the scratchpad contains the final result. Decode it to a word.
        auto result_vec = scratchpad.read("result");
        std::string reply = "";
        
        if (!result_vec.empty()) {
            reply = language.best_word(result_vec);
        } else {
            // If no clear result from reasoning, use fast inner speech (Imagination)
            auto think_res = think(1);
            if (!think_res.words.empty()) {
                reply = think_res.words[0];
            } else {
                reply = "...";
            }
        }
        
        return reply;
    }

    int force_reason_step(int op_idx, const std::string& goal_word) {
        auto goal_vec = language.encode(goal_word);
        // Use same fresh-slot context as direct_reason_step for train/infer consistency
        Scratchpad tmp = scratchpad;
        auto current_ctx = simulate_op(Op::HALT, tmp);
        std::vector<float> h, inp;
        auto [logits, value] = bg_controller.forward(current_ctx, goal_vec, h, inp);
        // Record trace for the FORCED op so reinforce_bg can update W2 row for that op
        bg_controller.record_trace(op_idx, h, inp, value);
        Op forced = (Op)op_idx;
        simulate_op(forced, scratchpad);
        auto next_ctx = simulate_op(Op::HALT, scratchpad);
        scratchpad.branch(next_ctx, 0.0f);
        scratchpad.move_to_best_child();
        return op_idx;
    }

    // Greedy one-shot op selection — mirrors training without PUCT tree overhead.
    // Records a trace so reinforce_bg can backprop gradient.
    int direct_reason_step(const std::string& goal_word) {
        auto goal_vec = language.encode(goal_word);
        // Use simulate_op(HALT) to get a fresh flat summary of ALL current slots,
        // including sensory_input that was just written by perceive(). This is the
        // key difference from current_tree_state() which returns a stale tree snapshot.
        Scratchpad tmp = scratchpad;
        auto current_ctx = simulate_op(Op::HALT, tmp);
        std::vector<float> h, inp;
        auto [logits, value] = bg_controller.forward(current_ctx, goal_vec, h, inp);
        // Apply emotion-state bias (soft tiebreaker, never overrides trained preferences)
        apply_emotion_bias(logits);
        auto probs = BasalGanglia::softmax(logits);
        int chosen = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        // Record trace so reinforce_bg has something to update
        bg_controller.record_trace(chosen, h, inp, value);
        // Apply op directly to scratchpad (no tree branching needed for linear parsing)
        simulate_op((Op)chosen, scratchpad);
        // Update tree state so next step sees the result
        auto next_ctx = simulate_op(Op::HALT, scratchpad);
        scratchpad.branch(next_ctx, 0.0f);
        scratchpad.move_to_best_child();
        return chosen;
    }

    // Apply a soft emotion-valence bias to BG logits.
    // Positive valence → approach ops (SPEAK, RETRIEVE, BIND_QUERY) get a small boost.
    // Negative valence → avoidant ops (HALT, COMPARE, NOT) get a small boost.
    // Scale 0.05 = gentle tiebreaker; never overrides trained preferences.
    void apply_emotion_bias(std::vector<float>& logits) const {
        float v = emotion.valence;        // [-1, +1]
        float a = emotion.arousal;        // [ 0,  1]
        float scale = 0.05f * (0.5f + 0.5f * a);  // stronger when aroused

        // Approach ops (positive valence boosts)
        static const int approach_ops[] = {
            (int)Op::SPEAK, (int)Op::SPEAK_SUBJ, (int)Op::SPEAK_REL,
            (int)Op::SPEAK_OBJ, (int)Op::RETRIEVE, (int)Op::BIND_QUERY
        };
        // Avoidant ops (negative valence boosts)
        static const int avoid_ops[] = {
            (int)Op::HALT, (int)Op::COMPARE, (int)Op::NOT
        };

        for (int op : approach_ops) {
            if (op < (int)logits.size()) logits[op] += scale * v;
        }
        for (int op : avoid_ops) {
            if (op < (int)logits.size()) logits[op] += scale * (-v);
        }
    }

    int reason_step(const std::string& goal_word, float epsilon = 0.0f) {
        auto goal_vec = language.encode(goal_word);
        auto current_ctx = scratchpad.current_tree_state();
        
        std::vector<float> h, inp;
        auto [logits, current_value] = bg_controller.forward(current_ctx, goal_vec, h, inp);
        
        struct Eval { int op; float cost; int child_id; };
        std::vector<Eval> evals;
        auto probs = BasalGanglia::softmax(logits);
        
        std::mutex evals_mtx;
        
        #pragma omp parallel for
        for (int op_idx = 0; op_idx < (int)Op::N_OPS; op_idx++) {
            if (op_idx == (int)Op::HALT) continue;
            if (probs[op_idx] < 0.05f) continue;
            
            Scratchpad sim_pad = scratchpad;
            auto next_ctx = simulate_op((Op)op_idx, sim_pad, false);
            std::vector<float> local_h, local_inp;
            auto [next_logits, next_val] = bg_controller.forward(next_ctx, goal_vec, local_h, local_inp);
            
            // PUCT style heuristic: combine Critic value with Actor prior
            // This prevents the search from falling into OOD hallucinations
            float c_puct = 2.0f;
            float h_cost = (1.0f - next_val) - c_puct * probs[op_idx];
            
            int child_id = scratchpad.branch(next_ctx, h_cost);
            
            std::lock_guard<std::mutex> lock(evals_mtx);
            evals.push_back({op_idx, h_cost, child_id});
        }
        
        if (evals.empty()) return (int)Op::HALT;
        
        int best_id = scratchpad.move_to_best_child();
        int picked_op = (int)Op::HALT;
        for (auto& e : evals) {
            if (e.child_id == best_id) {
                picked_op = e.op;
                break;
            }
        }
        
        bg_controller.record_trace(picked_op, h, inp, current_value);
        simulate_op((Op)picked_op, scratchpad);
        return picked_op;
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
