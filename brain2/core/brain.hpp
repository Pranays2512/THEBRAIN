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

#include <atomic>
#include "debug.hpp"
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
#include "logic_engine.hpp"
#include "policy_engine.hpp"

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
    bool  wm_passed;              // did it pass working memory gate?
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
    LogicEngine           logic_engine;
    // Crisp symbolic reasoner (the ported neurosymbolic executive): stored
    // policies + crisp numeric facts, solved/learned natively. Lives beside the
    // fuzzy binding memory — the crisp half of the brain.
    PolicyMemory          policy_mem;
    std::map<std::pair<std::string, std::string>, double> crisp_facts;
    size_t crisp_conflicts = 0;   // # of times a taught fact OVERWROTE a different value
    size_t oov_count = 0;         // # of out-of-vocabulary words given a random embedding
    size_t ep_seen_ = 0;          // tokens seen by the episodic min-rate floor

    // The crisp store has no per-fact uncertainty (single scalars). teach_fact still
    // overwrites, but a conflicting overwrite (same entity/rel, different value) is no
    // longer SILENT — it's counted + logged, so a Python caller poisoning the truth
    // store (audit: teach_fact bypasses the gate) is at least visible/auditable.
    void teach_fact(const std::string& entity, const std::string& rel, double value) {
        auto it = crisp_facts.find({entity, rel});
        if (it != crisp_facts.end() && std::abs(it->second - value) > 1e-9) {
            crisp_conflicts++;
            B2DEBUG("[crisp] CONFLICT %s.%s: %g -> %g (overwriting)\n",
                    entity.c_str(), rel.c_str(), it->second, value);
        }
        crisp_facts[{entity, rel}] = value;
    }
    void policy_add(const std::string& target, const std::vector<std::string>& inputs,
                    const ExprPtr& expr) {
        policy_mem.add(Policy{target, inputs, expr});
    }
    FactFn fact_fn() {
        return [this](const std::string& e, const std::string& r, double& out) -> bool {
            auto it = crisp_facts.find({e, r});
            if (it == crisp_facts.end()) return false;
            out = it->second; return true;
        };
    }
    std::optional<double> policy_solve(const std::string& entity, const std::string& rel) {
        PolicyEngine eng(&policy_mem, fact_fn(), true);
        return eng.solve(entity, rel);
    }
    bool policy_learn(const std::string& target, const std::string& entity) {
        PolicyEngine eng(&policy_mem, fact_fn(), true);
        return eng.learn(target, entity);
    }

    std::unordered_map<std::string, double> profile_times_;
    std::vector<std::string> spoken_words;

    // Replay buffer of recently-learned token sequences, for dream
    // consolidation. Each entry is (token embeddings, next-token target ids) —
    // exactly what the predictor trains on. Bounded ring; oldest evicted.
    struct ReplaySeq {
        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
    };
    std::deque<ReplaySeq> replay_buffer_;
    size_t replay_capacity_ = 300;

    // Automatic dream consolidation: rehearse buffered sequences every
    // `replay_interval_` training calls. Interleaving replay with new learning
    // is what cut catastrophic forgetting 84% (faithful, validated). On by
    // default; disable for clean single-corpus LM benchmarking.
    bool  auto_replay_      = true;
    int   replay_interval_  = 4;
    int   replay_samples_   = 4;
    long  train_calls_      = 0;

    void push_replay(std::vector<std::vector<float>> inputs, std::vector<int> targets) {
        if (inputs.empty()) return;
        replay_buffer_.push_back({std::move(inputs), std::move(targets)});
        while (replay_buffer_.size() > replay_capacity_) replay_buffer_.pop_front();
    }
    int replay_size() const { return (int)replay_buffer_.size(); }

    std::string get_profiling_report() const {
        std::string s = "--- Brain Profiling Report (microseconds) ---\n";
        for (const auto& [mod, time] : profile_times_) {
            s += mod + ": " + std::to_string(time) + " us\n";
        }
        return s;
    }

    std::vector<std::string> get_spoken_words() {
        return spoken_words;
    }
    
    void clear_spoken_words() {
        spoken_words.clear();
    }

private:
    std::unique_ptr<std::mutex> mtx_;
    int                         step_;
    std::vector<float>          prev_act_map_;
    std::vector<float>          prev_input_;   // buffered for 1-step-ahead prediction
    bool                        have_prev_act_ = false;
    std::vector<float>          last_act_map_;   // last SOM activation for grounding

    // σ-gating: online EWMA of CE errors for self-calibrating thresholds
    // WM gate fires at err > mu + 1*sigma (~16% rate), episodic at mu + 2*sigma (~2.5%)
    float err_mu_  = 0.f;   // running mean of CE
    float err_var_ = 1.f;   // running variance (initialized to 1 so sigma=1 early)
    int   err_n_   = 0;     // count of CE samples seen

public:
    void update_error_stats(float err) noexcept {
        // Welford online update with EWMA decay (alpha=0.01 per step)
        err_n_++;
        float alpha = (err_n_ < 100) ? (1.f / err_n_) : 0.01f;
        float delta = err - err_mu_;
        err_mu_ += alpha * delta;
        err_var_ = (1.f - alpha) * (err_var_ + alpha * delta * delta);
    }

    float wm_threshold()       const noexcept { return err_mu_ + std::sqrt(std::max(0.f, err_var_)); }
    float episodic_threshold() const noexcept { return err_mu_ + 2.f * std::sqrt(std::max(0.f, err_var_)); }

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
          // LSTM SGD lr: 0.001 measured stable; 0.005 oscillates once the
          // Adam-trained LM head starts feeding real gradients down.
          predictor(n_dims, hidden_dim, n_dims, 5, 0.001f, seed),
          episodic(som_rows * som_cols, episodic_max, 0.3f),
          working_mem(som_rows * som_cols, wm_capacity, 0.95f),
          language(n_dims, 0.05f),
          imagination(&predictor, 50),
          emotion(0.05f, 0.01f),
          attention(som_rows * som_cols, 0.1f, 0.3f),
          self_model(self_neurons, seed),
          symbolic(n_dims),
          scratchpad(n_dims),
          reasoning(&symbolic, n_dims, 50, 0.01f, &predictor),
          // V3 & V4 components
          pc_som(som_rows * som_cols, 0.05f, 0.01f),
          pc_hpred(som_rows * som_cols, 0.05f, 0.01f),
          pc_wm(som_rows * som_cols, 0.001f, 0.1f),
          pc_bg(1, 0.001f, 0.1f),
          binding(n_dims, 2000),    // fuzzy-recall cache cap; small ON PURPOSE — the exact
                                    // fact GRAPH keeps ALL facts and is what reasoning uses.
                                    // Raising this slows bulk ingest ~6x (256-bucket LSH fills);
                                    // the real fix for bigger fuzzy recall is more LSH planes.
          analogy(&binding),
          global_ws(som_rows * som_cols),
          bg_controller(n_dims, 0.001f, seed),
          procedures(n_dims),
          logic_engine(n_dims, language, symbolic, binding, episodic, som, working_mem, analogy, pc_wm, spoken_words),
          h_predictor(som_rows * som_cols, 128, 64, seed),
          step_(0),
          mtx_(std::make_unique<std::mutex>()) {
        symbolic.seed_math_symbols();
    }

    Brain(Brain&&)            = default;
    Brain& operator=(Brain&&) = delete;
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
        if (!episodic_path.empty()) {
            try {
                episodic = EpisodicMemory::load(episodic_path);
                B2DEBUG("DEBUG: Loaded episodic memory with %d episodes\n", episodic.episode_count());
            } catch (const std::exception& e) {
                printf("Warning: Could not load episodic memory: %s\n", e.what());
                // Silently instantiate empty episodic memory
                episodic = EpisodicMemory(n_dims, 2000, 0.3f);
            }
        }
        n_dims      = som.n_dims;
        som_rows    = som.rows;
        som_cols    = som.cols;
        emotion    = Emotion::load(emotion_path);
        self_model = SelfModel::load(self_path);
        symbolic   = Symbolic::load(symbolic_path);

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

        // crisp reasoner: load policies + facts from the save directory if present
        std::string dir = predictor_path.substr(0, predictor_path.find_last_of('/'));
        policy_mem = PolicyMemory();
        policy_mem.load(dir + "/policies.txt");
        std::ifstream ff(dir + "/facts.txt");
        std::string fe, fr; double fv;
        while (ff >> fe >> fr >> fv) crisp_facts[{fe, fr}] = fv;
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

        // crisp reasoner: stored policies + crisp facts
        policy_mem.save(directory + "/policies.txt");
        std::ofstream ff(directory + "/facts.txt");
        for (auto& kv : crisp_facts)
            ff << kv.first.first << '\t' << kv.first.second << '\t' << kv.second << '\n';
    }

    bool commit_episode(float err, const std::vector<float>& payload = {}) {
        return episodic.commit(err, payload);
    }

    void set_active_vocab(const std::vector<int>& active_indices) {
        predictor.set_active_vocab(language.flat_embeddings_ptr(), language.vocab_size(), active_indices, language.is_frozen_ptr());
    }

    // PERCEIVE: process one raw input vector through full pipeline
    PerceiveResult perceive(const std::vector<float>& input, int target_word_id = -1, ErrorMode error_mode = ErrorMode::FULL) {
        auto start_all = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(*mtx_);
        scratchpad.write("sensory_input", input, "sensory");

        // 1. SOM: find BMU + activation map
        auto t0 = std::chrono::high_resolution_clock::now();
        int bmu      = som.find_bmu(input);
        auto act_map = som.activation_map(input);
        som.update(input, bmu, 1.f + emotion.lr_modulator() * 0.5f);
        auto t1 = std::chrono::high_resolution_clock::now();
        profile_times_["SOM"] += std::chrono::duration<double, std::micro>(t1 - t0).count();
        
        // Add to Episodic Memory
        episodic.observe(act_map);

        // 2. Predictive coding 1: SOM output -> Error signal
        t0 = std::chrono::high_resolution_clock::now();
        auto pc1_err = pc_som.propagate(act_map);
        bool do_propagate = pc_som.should_propagate();
        pc_som.update();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["PC_SOM"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 3. Predictor (Softmax LM)
        t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<float, int>> pred_out;
        
        if (have_prev_act_) {
            pred_out = predictor.step(prev_input_, target_word_id, error_mode);
        } else {
            pred_out = predictor.step(input, -1);
        }
        prev_input_ = input;
        
        // Convert LM output Top-K words back to a sparse SOM activation map
        std::vector<float> pred_next(som.n_neurons, 0.f);
        for(const auto& pair : pred_out) {
            float prob = pair.first;
            int wid = pair.second;
            if (prob > 0.05f) { // Only map confident possibilities
                const float* emb = language.flat_embeddings_ptr() + wid * predictor.target_dim;
                std::vector<float> coord(emb, emb + predictor.target_dim);
                int pred_bmu = som.find_bmu(coord);
                if (pred_bmu >= 0 && pred_bmu < som.n_neurons) {
                    pred_next[pred_bmu] += prob; 
                }
            }
        }
        
        // Normalize pred_next if max > 1.0
        float max_pred = 0.f;
        for(float v : pred_next) if(v > max_pred) max_pred = v;
        if(max_pred > 1.f) {
            for(float& v : pred_next) v /= max_pred;
        }

        prev_act_map_  = pc1_err;
        last_act_map_  = act_map;
        have_prev_act_ = true;
        float error = predictor.last_error();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Predictor"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 4. Hierarchical predictor (Processes PC1 Error)
        t0 = std::chrono::high_resolution_clock::now();
        if (do_propagate) {
            h_predictor.observe(pc1_err);
        }
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["H_Predictor"] += std::chrono::duration<double, std::micro>(t1 - t0).count();
        
        // 5. Predictive coding 2: H-Predictor -> WM
        t0 = std::chrono::high_resolution_clock::now();
        auto pc2_err = pc_hpred.propagate(pc1_err);
        pc_hpred.update();
        bool propagate_wm = pc_hpred.should_propagate();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["PC_HPRED"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 6. Global Workspace competition
        t0 = std::chrono::high_resolution_clock::now();
        global_ws.bid((int)GWModule::SOM,    pc_som.error_norm, pc2_err);
        global_ws.bid((int)GWModule::PREDICT, 1.f - error,      pred_next);
        global_ws.bid((int)GWModule::EMOTION, emotion.salience(), act_map);
        int gw_winner = global_ws.compete();
        scratchpad.write("attention", global_ws.broadcast(), "gw");
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Global_Workspace"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 7. Emotion
        t0 = std::chrono::high_resolution_clock::now();
        emotion.from_prediction_error(error);
        emotion.tick();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Emotion"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 8. Attention & Working Memory (Process PC2 Error)
        t0 = std::chrono::high_resolution_clock::now();
        auto attn_result = attention.gate(pc2_err, error, emotion.attention_modulator());
        bool wm_gate_open = attn_result.passed &&
            (global_ws.is_winner((int)GWModule::SOM) || propagate_wm || error > predictor.surprise_threshold());
        if (wm_gate_open) {
            working_mem.gate(pc2_err, emotion.salience());
        }
        working_mem.tick();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Working_Memory"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 9. Predictive coding 3: WM -> Episodic/Binding
        t0 = std::chrono::high_resolution_clock::now();
        auto ctx_summary = working_mem.context();
        if (ctx_summary.empty()) ctx_summary.assign(som.n_neurons, 0.f);
        auto pc3_err = pc_wm.propagate(ctx_summary);
        
        // CAUSAL LEARNING: If WM violates predictions (Surprise!), violently unlearn the context!
        if (pc_wm.should_propagate()) {
            working_mem.disrupt_by_error(pc3_err);
        }
        
        pc_wm.update();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["PC_WM"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 10. EpisodicMemory (Process PC3 Error)
        t0 = std::chrono::high_resolution_clock::now();
        bool episodic_active = global_ws.is_winner((int)GWModule::SOM) ||
                               global_ws.is_winner((int)GWModule::EMOTION);
        if (episodic_active && pc_wm.should_propagate()) {
            episodic.observe(pc3_err);
        }
        episodic.surprise_threshold = episodic_threshold();
        bool stored = episodic_active && episodic.commit(error);
        
        static std::atomic<int> word_count{0};
        static std::atomic<int> wm_open_count{0};
        static std::atomic<int> ep_commit_count{0};
        if (word_count.load() < 500) {
            int wc = ++word_count;
            if (wm_gate_open) wm_open_count++;
            if (stored) ep_commit_count++;
            if (wc == 500) {
                B2DEBUG("[Instrumentation] First 500 words: WM gate opened %.1f%%, Episodic committed %.1f%%\n",
                        (float)wm_open_count.load() / 5.0f, (float)ep_commit_count.load() / 5.0f);
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Episodic_Memory"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 11. Predictive coding 4: WM Context -> BG Controller
        t0 = std::chrono::high_resolution_clock::now();
        auto pc4_err = pc_bg.propagate(ctx_summary);
        pc_bg.update();
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["PC_BG"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 12. Basal Ganglia (Process PC4 Error)
        t0 = std::chrono::high_resolution_clock::now();
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
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Basal_Ganglia"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // 13. SelfModel
        t0 = std::chrono::high_resolution_clock::now();
        auto istate = build_internal_state();
        self_model.observe(istate);
        t1 = std::chrono::high_resolution_clock::now();
        profile_times_["Self_Model"] += std::chrono::duration<double, std::micro>(t1 - t0).count();

        step_++;

        auto end_all = std::chrono::high_resolution_clock::now();
        profile_times_["Total_Perceive"] += std::chrono::duration<double, std::micro>(end_all - start_all).count();

        PerceiveResult r;
        r.bmu              = bmu;
        r.prediction_error = error;
        r.attention_passed = attn_result.passed;
        r.wm_passed        = wm_gate_open;
        r.episodic_stored  = stored;
        r.valence          = emotion.valence;
        r.arousal          = emotion.arousal;
        r.salience         = attn_result.score;
        r.self_concept     = self_model.current_concept(istate);
        return r;
    }

    // Fast sequential LM batch training path
    float train_lm_sequence(const std::string& text) {
        std::istringstream iss(text);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) words.push_back(word);

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;

        for (size_t i = 0; i < words.size(); i++) {
            if (language.knows(words[i])) {
                inputs.push_back(language.encode(words[i]));
                int target_id = -1;
                if (i + 1 < words.size() && language.knows(words[i + 1])) {
                    target_id = language.word_id(words[i + 1]);
                }
                targets.push_back(target_id);
            }
        }
        
        if (inputs.empty()) return 0.f;
        return predictor.train_lm_sequence(inputs, targets);
    }

    // Offline evaluation: total negative log-likelihood (nats) and the number
    // of scored tokens over `text`. Lets callers compute tokenizer-invariant
    // metrics (bits/char, NLL/word) instead of per-token mean CE, which is not
    // comparable across tokenizations. Does not update any weights.
    std::pair<double, long> eval_text_nll(const std::string& text) {
        std::istringstream iss(text);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) words.push_back(word);

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        for (size_t i = 0; i < words.size(); i++) {
            if (!language.knows(words[i])) continue;
            inputs.push_back(language.encode(words[i]));
            int target_id = (i + 1 < words.size() && language.knows(words[i + 1]))
                          ? language.word_id(words[i + 1]) : -1;
            targets.push_back(target_id);
        }
        if (inputs.empty()) return {0.0, 0};

        bool was_offline = predictor.is_offline();
        predictor.set_offline(true);
        std::vector<float> ce;
        predictor.train_lm_sequence_per_token(inputs, targets, ce);
        predictor.set_offline(was_offline);

        double total = 0.0;
        long n = 0;
        for (float c : ce) if (c >= 0.f) { total += c; n++; }
        return {total, n};
    }

    // Fused LM training + cognitive perception in one pass per segment.
    // - Runs train_lm_sequence_per_token to get real per-position CE (one LSTM forward+backward)
    // - For each word feeds that real CE through the cognitive pipeline
    // - Uses σ-gated thresholds (self-calibrating) instead of hardcoded magic numbers
    // - One SOM scan per word (find_bmu_and_activate) instead of three separate scans
    // Returns mean CE loss.
    float train_lm_sequence_fused(const std::string& text) {
        std::istringstream iss(text);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) words.push_back(word);

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        std::vector<int> word_indices; // maps inputs[i] → words[j]

        for (size_t i = 0; i < words.size(); i++) {
            if (language.knows(words[i])) {
                inputs.push_back(language.encode(words[i]));
                int target_id = -1;
                if (i + 1 < words.size() && language.knows(words[i + 1]))
                    target_id = language.word_id(words[i + 1]);
                targets.push_back(target_id);
                word_indices.push_back((int)i);
            }
        }

        if (inputs.empty()) return 0.f;

        // LM training pass — gets per-token CE for real
        std::vector<float> ce_per_token;
        float mean_ce = predictor.train_lm_sequence_per_token(inputs, targets, ce_per_token);

        // Buffer this sequence for dream-time replay consolidation.
        push_replay(inputs, targets);

        // Cognitive pass — one SOM scan per word, σ-gated
        std::lock_guard<std::mutex> lock(*mtx_);
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto& inp = inputs[i];
            float ce = ce_per_token[i]; // -1 if no valid target
            if (ce < 0.f) ce = err_mu_; // treat unknown-target words as average surprise

            // Update σ-gating statistics
            update_error_stats(ce);

            // SOM: single distance scan for BMU + activation map
            auto [bmu, act_map] = som.find_bmu_and_activate(inp);
            // PC error -> SOM (the representation now LEARNS from being wrong): scale the
            // SOM update by how surprising this token was, z-scored against the running
            // baseline. Surprising input -> larger weight move (re-represent it); familiar
            // input -> smaller. Previously the error reached the SOM only via emotion.
            float sigma = std::sqrt(std::max(1e-6f, err_var_));
            float surprise_mod = std::max(-0.5f, std::min((ce - err_mu_) / sigma, 2.f)) * 0.3f;
            som.update(inp, bmu, 1.f + emotion.lr_modulator() * 0.5f + surprise_mod);

            episodic.observe(act_map);

            // Predictive coding on SOM
            auto pc1_err = pc_som.propagate(act_map);
            bool do_propagate = pc_som.should_propagate();
            pc_som.update();

            // H-Predictor
            if (do_propagate) h_predictor.observe(pc1_err);
            auto pc2_err = pc_hpred.propagate(pc1_err);
            pc_hpred.update();
            bool propagate_wm = pc_hpred.should_propagate();

            // Global Workspace
            global_ws.bid((int)GWModule::SOM,    pc_som.error_norm, pc2_err);
            global_ws.bid((int)GWModule::PREDICT, 1.f - ce / (ce + 1.f), act_map);
            global_ws.bid((int)GWModule::EMOTION, emotion.salience(), act_map);
            int gw_winner = global_ws.compete();
            (void)gw_winner;

            // Emotion from real CE
            emotion.from_prediction_error(ce);
            emotion.tick();

            // σ-gated WM: fires ~16% of words
            float wm_thr = wm_threshold();
            bool wm_gate_open = ce > wm_thr || propagate_wm ||
                                global_ws.is_winner((int)GWModule::SOM);
            if (wm_gate_open) working_mem.gate(pc2_err, emotion.salience());
            working_mem.tick();

            // PC3 on WM context
            auto ctx_summary = working_mem.context();
            if (ctx_summary.empty()) ctx_summary.assign(som.n_neurons, 0.f);
            auto pc3_err = pc_wm.propagate(ctx_summary);
            if (pc_wm.should_propagate()) working_mem.disrupt_by_error(pc3_err);
            pc_wm.update();

            // σ-gated Episodic: fires ~2.5% of words
            float ep_thr = episodic_threshold();
            bool episodic_active = global_ws.is_winner((int)GWModule::SOM) ||
                                   global_ws.is_winner((int)GWModule::EMOTION);
            if (episodic_active && pc_wm.should_propagate()) episodic.observe(pc3_err);
            episodic.surprise_threshold = ep_thr;
            // Episodic was triple-gated on surprise (ce > ep_thr): a converged model stops
            // storing familiar domains exactly when re-examination matters. Add a min-rate
            // floor so ~2% of familiar tokens still get committed.
            bool min_rate = (++ep_seen_ % 50 == 0);
            bool stored = episodic_active && (ce > ep_thr || min_rate) && episodic.commit(ce);

            // Instrumentation (first 500 words)
            static std::atomic<int> word_count_f{0};
            static std::atomic<int> wm_open_count_f{0};
            static std::atomic<int> ep_commit_count_f{0};
            if (word_count_f.load() < 500) {
                int wcf = ++word_count_f;
                if (wm_gate_open) wm_open_count_f++;
                if (stored) ep_commit_count_f++;
                if (wcf == 500) {
                    B2DEBUG("[Fused] First 500 words: WM gate %.1f%%, Episodic %.1f%%, "
                            "CE_mu=%.2f, CE_sigma=%.2f, wm_thr=%.2f, ep_thr=%.2f\n",
                            (float)wm_open_count_f.load() / 5.f,
                            (float)ep_commit_count_f.load() / 5.f,
                            err_mu_, std::sqrt(std::max(0.f, err_var_)),
                            wm_threshold(), episodic_threshold());
                }
            }
            (void)stored;

            step_++;
        }

        have_prev_act_ = false; // reset sequence boundary

        // Automatic dream consolidation: rehearse buffered sequences while
        // learning, every replay_interval_ calls. This is the interleaved
        // faithful replay that cut catastrophic forgetting 84%.
        if (auto_replay_ && ++train_calls_ % replay_interval_ == 0 &&
            (int)replay_buffer_.size() > replay_samples_) {
            dream_replay_faithful(replay_samples_, 1);
        }
        return mean_ce;
    }

    void perceive_text(const std::string& text, ErrorMode error_mode = ErrorMode::FULL) {
        std::string current_word;
        auto process_word = [&](const std::string& w) {
            if (w.empty()) return;
            if (!language.knows(w)) {
                oov_count++;                    // OOV gets a random embedding -> flag it, don't hide it
                B2DEBUG("[lang] OOV '%s' -> random embedding (may false-match by chance)\n", w.c_str());
                language.register_word(w);
                symbolic.bind(w);
            }
            int wid = language.word_id(w);
            perceive(language.encode(w), wid, error_mode);
        };
        
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                process_word(current_word);
                current_word.clear();
            } else {
                current_word += c;
            }
        }
        process_word(current_word);
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
            
            // Train predictor: disabled for MDN during daydreaming (requires 128-D coords)
            // predictor.step(prev_err, &curr_err);
            
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
        // Enforce dense embedding space; do not mix sparse SOM maps
        auto ctx = std::vector<float>(n_dims, 0.f); 
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
        episodic.clear_buffer();
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
        std::vector<float> current_vec = prev_input_;
        if (current_vec.empty()) current_vec = std::vector<float>(n_dims, 0.f);

        // Record initial state for legacy concepts tracking
        std::vector<float> init_act(som.n_neurons, 0.f);
        int init_bmu = som.find_bmu(current_vec);
        if (init_bmu >= 0 && init_bmu < som.n_neurons) init_act[init_bmu] = 1.0f;
        result.concepts.push_back(init_act);

        // DO NOT reset predictor here. We want to continue the train of thought from the perceived prompt!
        predictor.set_offline(true);

        for (int i = 0; i < steps; i++) {
            auto pred_out = predictor.step(current_vec);
            if (pred_out.empty()) break;
            int best_wid = pred_out[0].second;
            std::vector<float> next_coord = predictor.get_embedding(best_wid);

            // Decode this 128-D coordinate to the nearest word
            auto word = language.best_word(next_coord, result.words);
            if (!word.empty()) {
                result.words.push_back(word);
                // Inner speech feeds back into working memory
                auto word_vec = language.encode(word);
                working_mem.gate(word_vec, 0.f);
                working_mem.tick();
                current_vec = word_vec; // Use the actual word's vector for the next step
            } else {
                current_vec = next_coord; // Fallback to raw predicted coordinate
            }

            // Create a sparse activation map for legacy 'concepts' tracking
            std::vector<float> next_act(som.n_neurons, 0.f);
            int pred_bmu = som.find_bmu(current_vec);
            if (pred_bmu >= 0 && pred_bmu < som.n_neurons) next_act[pred_bmu] = 1.0f;
            result.concepts.push_back(next_act);
        }

        predictor.set_offline(false);
        // Honest coherence: mean cosine similarity between consecutive inner-speech
        // words (a smooth, on-topic trajectory scores high; a jumpy one low). Was a
        // hardcoded 1.0 that made any downstream quality gate meaningless.
        result.coherence = 0.0f;
        {
            float total = 0.f; int pairs = 0;
            std::vector<float> prev;
            for (const auto& w : result.words) {
                std::vector<float> v = language.encode(w);
                if (!prev.empty() && v.size() == prev.size()) {
                    float dot = 0.f, na = 0.f, nb = 0.f;
                    for (size_t k = 0; k < v.size(); k++) {
                        dot += v[k] * prev[k]; na += v[k] * v[k]; nb += prev[k] * prev[k];
                    }
                    if (na > 1e-9f && nb > 1e-9f) {
                        total += dot / (std::sqrt(na) * std::sqrt(nb)); pairs++;
                    }
                }
                prev = std::move(v);
            }
            if (pairs) result.coherence = total / pairs;
        }
        return result;
    }

    std::vector<float> simulate_op(Op op, Scratchpad& pad, bool commit = true) {
        return logic_engine.execute_op(op, pad, commit);
    }

    void start_reasoning() {
        bg_controller.clear_traces();

        // Skip PC Warm-up during RL training to massively speed up iteration!
        // The PC model is frozen during RL anyway.

        std::vector<float> initial_ctx = simulate_op(Op::HALT, scratchpad);
        scratchpad.start_tree(initial_ctx);
    }

    std::vector<int> reason(const std::string& goal_word, int max_steps = 10, float epsilon = 0.0f) {
        std::vector<int> solution_path;
        auto goal_vec = language.encode(goal_word);
        
        std::vector<float> initial_ctx = simulate_op(Op::HALT, scratchpad); // just gets current summary
        
        // --- Procedural Memory Fast-Path ---
        auto* proc = procedures.retrieve(initial_ctx);
        if (proc && !proc->steps.empty()) {
            for (auto step_op : proc->steps) {
                solution_path.push_back((int)step_op);
                simulate_op(step_op, scratchpad);
            }
            return solution_path;
        }
        
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
            
            // --- Strict Top-K Pruning ---
            std::vector<std::pair<float, int>> ranked_ops;
            for (int op_idx = 0; op_idx < (int)Op::N_OPS; op_idx++) {
                if (op_idx == (int)Op::HALT) continue;
                ranked_ops.push_back({probs[op_idx], op_idx});
            }
            std::sort(ranked_ops.begin(), ranked_ops.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            
            int top_k = 3;
            for (int i = 0; i < top_k && i < (int)ranked_ops.size(); i++) {
                int op_idx = ranked_ops[i].second;
                float prob = ranked_ops[i].first;
                
                if (prob < 0.01f) continue;
                
                auto sim_pad = scratchpad; // Deep copy
                auto next_ctx = simulate_op((Op)op_idx, sim_pad, false);
                
                std::vector<float> h_child, inp_child;
                auto [logits_c, value_c] = bg_controller.forward(next_ctx, goal_vec, h_child, inp_child);
                
                // PUCT style heuristic: combine Critic value with Actor prior
                float c_puct = 2.0f;
                float h_cost = (1.0f - value_c) - c_puct * prob;
                
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

    std::string cognitive_step(const std::string& input_text) {
        // 1. PERCEIVE (tokenize properly)
        perceive_text(input_text);
        
        // 2. THINK (Reasoning)
        // Load working memory context into scratchpad to ground the tree search
        auto ctx = working_mem.context();
        if (!ctx.empty()) {
            // Write the full 65536D spatial map directly to scratchpad for high-fidelity retrieval
            scratchpad.write("context_map", ctx, "context");
            
            // Legacy BMU extraction for dense 128D operations
            int bmu = 0; float max_val = -1.f;
            for (int i = 0; i < (int)ctx.size(); i++) {
                if (ctx[i] > max_val) { max_val = ctx[i]; bmu = i; }
            }
            scratchpad.write("subject", som.neuron_weights(bmu), "context");
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


    void load_bg(const std::string& path) {
        bg_controller.load(path);
    }
    void save_bg(const std::string& path) const {
        bg_controller.save(path);
    }

    int force_reason_step(int op_idx, const std::string& goal_word) {
        fprintf(stderr, "DEBUG force_reason_step: wm.empty=%d, pad.has(context_map)=%d\\n", 
                working_mem.empty(), scratchpad.has("context_map"));
                
        if (!scratchpad.has("context_map")) {
            auto ctx_act_map = episodic.current_summary();
            scratchpad.write("context_map", ctx_act_map, "context");
            
            // For legacy subject tracking, just write the current working memory context
            auto ctx = working_mem.context();
            int bmu = 0; float max_val = -1.f;
            for (int i = 0; i < (int)ctx.size(); i++) {
                if (ctx[i] > max_val) { max_val = ctx[i]; bmu = i; }
            }
            scratchpad.write("subject", som.neuron_weights(bmu), "context");
        }

        auto goal_vec = language.encode(goal_word);
        
        // Use exactly the same interleaved state representation from Scratchpad!
        auto current_ctx = scratchpad.current_tree_state();
        
        std::vector<float> h, inp;
        auto [logits, value] = bg_controller.forward(current_ctx, goal_vec, h, inp);
        
        // Record trace for the FORCED op so reinforce_bg can update W2 row for that op
        bg_controller.record_trace(op_idx, h, inp, value);
        
        // Execute the actual operation so that scratchpad is updated for evaluation
        Op op = static_cast<Op>(op_idx);
        fprintf(stderr, "DEBUG: force_reason_step calling execute_op with op_idx=%d\\n", op_idx);
        logic_engine.execute_op(op, scratchpad);

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

    // ── Dream consolidation via replay ──────────────────────────────────────
    // FAITHFUL replay: re-train the predictor on a sample of the exact
    // sequences it recently learned. This is experience replay — the standard
    // defense against catastrophic forgetting. Returns mean replay CE.
    float dream_replay_faithful(int n_samples = 16, int passes = 1) {
        if (replay_buffer_.empty()) return 0.f;
        std::mt19937 rng(step_ + 1234567u);
        std::uniform_int_distribution<int> pick(0, (int)replay_buffer_.size() - 1);
        float total = 0.f; int cnt = 0;
        for (int p = 0; p < passes; p++) {
            for (int s = 0; s < n_samples; s++) {
                const auto& seq = replay_buffer_[pick(rng)];
                predictor.reset();  // clean state per replayed sequence
                total += predictor.train_lm_sequence(seq.inputs, seq.targets);
                cnt++;
            }
        }
        predictor.reset();
        return cnt ? total / cnt : 0.f;
    }

    // GENERATIVE replay: prime the predictor with a buffered sequence's first
    // token, let the model DREAM a continuation by sampling its own top-k
    // predictions, then train on that self-generated sequence at reduced lr.
    // More brain-like (REM-style novel recombination) and, unlike faithful
    // replay, consolidates the model's learned *structure* rather than verbatim
    // examples — at the risk of reinforcing its own drift. The A/B forgetting
    // number decides which is better. Returns mean generated CE.
    float dream_replay_generative(int n_samples = 16, int gen_len = 24,
                                  float lr_scale = 0.5f, int top_k = 5) {
        if (replay_buffer_.empty()) return 0.f;
        std::mt19937 rng(step_ + 7654321u);
        std::uniform_int_distribution<int> pick(0, (int)replay_buffer_.size() - 1);
        float saved_lr = predictor.lr();
        float total = 0.f; int cnt = 0;

        for (int s = 0; s < n_samples; s++) {
            const auto& seq = replay_buffer_[pick(rng)];
            if (seq.inputs.empty()) continue;

            // 1. Dream a sequence by sampling the model's own predictions.
            std::vector<std::vector<float>> gen_in;
            std::vector<int> gen_tgt;
            predictor.set_offline(true);
            predictor.reset();
            std::vector<float> cur = seq.inputs[0];
            for (int t = 0; t < gen_len; t++) {
                auto ranked = predictor.step(cur);   // offline forward, top-K
                if (ranked.empty()) break;
                int kk = std::min((int)ranked.size(), top_k);
                std::uniform_int_distribution<int> samp(0, kk - 1);
                int wid = ranked[samp(rng)].second;  // sample among top-k
                gen_in.push_back(cur);
                gen_tgt.push_back(wid);
                cur = predictor.get_embedding(wid);
                if (cur.empty()) break;
            }
            predictor.set_offline(false);

            // 2. Train on the dreamed sequence at reduced lr.
            if (gen_in.size() > 1) {
                predictor.reset();  // clean state per dreamed sequence
                predictor.set_lr(saved_lr * lr_scale);
                total += predictor.train_lm_sequence(gen_in, gen_tgt);
                predictor.set_lr(saved_lr);
                cnt++;
            }
        }
        predictor.reset();
        return cnt ? total / cnt : 0.f;
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

    void expand_dims(int new_dims) {
        if (new_dims <= n_dims) return;
        
        som.expand_dims(new_dims);
        predictor.expand_dims(new_dims);
        episodic.expand_dims(new_dims);
        working_mem.expand_dims(new_dims);
        language.expand_dims(new_dims);
        imagination.expand_dims(new_dims);
        symbolic.expand_dims(new_dims);
        scratchpad.expand_dims(new_dims);
        reasoning.expand_dims(new_dims);
        
        pc_som.expand_dims(new_dims);
        pc_hpred.expand_dims(new_dims);
        pc_wm.expand_dims(new_dims);
        pc_bg.expand_dims(new_dims);
        binding.expand_dims(new_dims);
        global_ws.expand_dims(new_dims);
        bg_controller.expand_dims(new_dims);
        procedures.expand_dims(new_dims);
        h_predictor.expand_dims(new_dims);
        logic_engine.expand_dims(new_dims);
        
        n_dims = new_dims;
    }
};

} // namespace brain2
