#pragma once
/*
 * cognitive_bridge.hpp — Full Integration: Crisp Engines ↔ Brain Cognitive Loop
 *
 * COMPLETE INFORMATION FLOW (every step logged):
 *
 *  Input Problem
 *      │
 *      ▼ encode_problem()  [language.encode + structural features]
 *  float vector [n_dims]
 *      │
 *      ▼ brain.perceive()  [LOGGED: all 13 stages below]
 *      ├─ [SOM]            BMU + novelty distance logged
 *      ├─ [PC-SOM]         Predictive coding error logged
 *      ├─ [Predictor]      LM prediction error logged
 *      ├─ [H-Predictor]    Hierarchical prediction logged
 *      ├─ [GlobalWS]       Winner module logged
 *      ├─ [Emotion]        valence / arousal before+after logged
 *      ├─ [Attention]      gate passed/failed + novelty score logged
 *      ├─ [WorkingMemory]  load + slot count logged
 *      ├─ [EpisodicMem]    stored/not + surprise threshold logged
 *      ├─ [BasalGanglia]   selected op logged
 *      └─ [SelfModel]      current concept logged
 *      │
 *      ▼ [IMAGINATION]     "What if I solved this?" — offline sim from WM context
 *      │  coherence / valence / frames logged
 *      │
 *      ▼ UnifiedProposer.solve()   [already logged by BrainLog]
 *      │  policy routing + confidence + fallback all logged
 *      │
 *      ▼ [EMOTION FEEDBACK] success/fail → valence nudge logged
 *      │
 *      ▼ brain.perceive(answer)    [re-perception of own answer]
 *      │
 *      ▼ brain.tick()              [drain daydream if pending]
 *      │  [DAYDREAM] triggered/not + dream frames + coherence logged
 *      │
 *      ▼ brain.think()             [inner speech: WM → words]
 *      │  words emitted + coherence logged
 *      │
 *      ▼ brain.learn_from_crisp()  [write result to BindingMemory]
 *      │
 *      ▼ CognitiveSolveResult returned
 */

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "fuzzy/core/brain.hpp"
#include "fuzzy/engines/synthesis/unified_proposer.hpp"

namespace brain3 {

using engines::synthesis::Problem;
using engines::synthesis::UnifiedProposer;
using engines::synthesis::BrainLog;
using engines::synthesis::brain_log;

// ── Structured Cognitive Solve Result ────────────────────────────────────────
struct CognitiveSolveResult {
    bool        solved;
    std::string problem_type;
    std::string policy_used;

    // Perception layer
    int         som_bmu;
    float       som_novelty;          // BMU distance — higher = more novel to the Brain
    float       prediction_error;     // how surprising was this problem

    // Cognitive state
    bool        attention_passed;
    bool        wm_gate_open;
    bool        episodic_stored;
    float       emotion_valence_before;
    float       emotion_arousal_before;
    float       emotion_valence_after;
    float       emotion_arousal_after;

    // Imagination
    bool        imagination_ran;
    float       imagination_coherence;
    float       imagination_valence;
    int         imagination_frames;

    // Inner speech
    std::vector<std::string> inner_speech_words;
    float       inner_speech_coherence;

    // Daydream
    bool        daydream_triggered;
};

// ── CognitiveBridge ────────────────────────────────────────────────────────────
class CognitiveBridge {
private:
    brain2::Brain& brain_;
    UnifiedProposer proposer_;

    // Solve cycle counter for periodic daydream
    int solve_count_ = 0;

    // ── Logging helpers ────────────────────────────────────────────────────────
    using L = BrainLog;

    void log(L::Level lvl, const std::string& comp, const std::string& msg) {
        brain_log().log(lvl, comp, msg);
    }
    void info(const std::string& comp, const std::string& msg)  { log(L::INFO,  comp, msg); }
    void warn(const std::string& comp, const std::string& msg)  { log(L::WARN,  comp, msg); }
    void debug(const std::string& comp, const std::string& msg) { log(L::DEBUG, comp, msg); }

    template<typename T>
    std::string f2(T v) {
        std::ostringstream o; o << std::fixed << std::setprecision(4) << v; return o.str();
    }
    template<typename T>
    std::string f1(T v) {
        std::ostringstream o; o << std::fixed << std::setprecision(1) << v * 100.0 << "%"; return o.str();
    }

    // ── Encode problem string → float vector ──────────────────────────────────
    std::vector<float> encode_problem(const Problem& p) {
        auto type_vec = brain_.language.encode(p.type);

        std::vector<float> data_vec(brain_.n_dims, 0.f);
        if (!p.data_str.empty()) {
            std::vector<std::string> tokens;
            std::string cur;
            for (char c : p.data_str) {
                if (std::isalnum(c) || c == '.' || c == '_') {
                    cur += c;
                } else {
                    if (!cur.empty()) { tokens.push_back(cur); cur = ""; }
                    tokens.push_back(std::string(1, c));
                }
            }
            if (!cur.empty()) tokens.push_back(cur);

            for (const auto& tok : tokens) {
                if (tok.empty() || tok == " ") continue;
                auto tv = brain_.language.encode(tok);
                for (int i = 0; i < brain_.n_dims && i < (int)tv.size(); ++i)
                    data_vec[i] += tv[i] / std::max(1, (int)tokens.size());
            }
        }

        // Blend: 70% type + 30% data
        std::vector<float> blended(brain_.n_dims, 0.f);
        for (int i = 0; i < brain_.n_dims; ++i)
            blended[i] = 0.7f * (i < (int)type_vec.size() ? type_vec[i] : 0.f)
                       + 0.3f * data_vec[i];

        // Structural feature slots at end of vector
        int nd = brain_.n_dims;
        if (nd > 8) {
            blended[nd-1] += (p.data_str.find('=') != std::string::npos) ? 0.5f : 0.f;
            blended[nd-2] += p.variables.empty() ? 0.f : std::min(0.5f, (float)p.variables.size() / 10.f);
            blended[nd-3] += p.test_fn   ? 0.5f : 0.f;
            blended[nd-4] += p.knowns.empty() ? 0.f : 0.5f;
        }
        return blended;
    }

    std::vector<float> encode_result(const std::string& label, bool success) {
        auto vec = brain_.language.encode(label.empty() ? "unknown" : label);
        if (!vec.empty()) vec.back() = success ? 1.0f : -1.0f;
        return vec;
    }

    // ── Log PerceiveResult details ─────────────────────────────────────────────
    void log_perceive(const std::string& context, const brain2::PerceiveResult& pr, float novelty) {
        std::ostringstream o;
        o << context << " │ BMU=" << pr.bmu
          << " novelty=" << f2(novelty)
          << " pred_err=" << f2(pr.prediction_error)
          << " attn=" << (pr.attention_passed ? "PASS" : "block")
          << " wm=" << (pr.wm_passed ? "OPEN" : "closed")
          << " epis=" << (pr.episodic_stored ? "STORED" : "no")
          << " val=" << f2(pr.valence)
          << " aro=" << f2(pr.arousal);
        info("Perceive", o.str());
    }

    // ── Compute SOM novelty (BMU distance before update) ──────────────────────
    float som_novelty(const std::vector<float>& vec) {
        int bmu = brain_.som.find_bmu(vec);
        auto bmu_w = brain_.som.neuron_weights(bmu);
        float dist = 0.f;
        for (int i = 0; i < brain_.n_dims && i < (int)bmu_w.size(); ++i) {
            float d = vec[i] - bmu_w[i];
            dist += d * d;
        }
        return std::sqrt(dist);
    }

    // ── Run Imagination: "what if I could solve this?" ────────────────────────
    brain2::Simulation run_imagination(const Problem& p, bool log_detail) {
        // Seed with current WM context so imagination extends the current thought
        auto wm_ctx = brain_.working_mem.context();
        if (wm_ctx.empty()) wm_ctx.assign(brain_.som.n_neurons, 0.f);

        // Simulate 8 steps forward from current WM state
        auto sim = brain_.imagination.simulate(wm_ctx, 8);

        if (log_detail) {
            std::ostringstream o;
            o << "problem='" << p.type << "' │ "
              << "frames=" << sim.frames.size()
              << " coherence=" << f2(sim.coherence)
              << " valence=" << f2(sim.valence)
              << " completed=" << (sim.completed ? "yes" : "no");
            if (sim.coherence > 0.6f) {
                o << " [COHERENT — insight possible]";
            } else if (sim.coherence < 0.3f) {
                o << " [INCOHERENT — high uncertainty]";
            }
            info("Imagination", o.str());
        }
        return sim;
    }

    // ── Run Think: inner speech after solving ─────────────────────────────────
    brain2::ThinkResult run_think(int steps, bool log_detail) {
        auto tr = brain_.think(steps);
        if (log_detail) {
            std::ostringstream o;
            o << "steps=" << steps
              << " words=[";
            for (size_t i = 0; i < tr.words.size(); ++i) {
                if (i) o << ", ";
                o << tr.words[i];
            }
            o << "] coherence=" << f2(tr.coherence);
            if (tr.coherence > 0.5f) {
                o << " [on-topic inner speech]";
            } else {
                o << " [scattered / low-confidence]";
            }
            info("InnerSpeech", o.str());
        }
        return tr;
    }

    // ── Log Daydream trigger ──────────────────────────────────────────────────
    bool maybe_daydream(bool log_it) {
        // tick() drains the pending_daydream_ flag set by perceive()'s InternalRouter
        // We log whether daydream was actually triggered
        bool was_pending = brain_.tick();
        if (log_it) {
            if (was_pending) {
                info("Daydream", "▶ Triggered (CONSOLIDATE mode — predictor replay on dream frames)");
            } else {
                debug("Daydream", "Not triggered this cycle");
            }
        }
        return was_pending;
    }

    // ── Log full brain state ───────────────────────────────────────────────────
    void log_brain_state(const std::string& tag) {
        auto is = brain_.build_internal_state();
        std::ostringstream o;
        o << tag
          << " │ valence=" << f2(is.valence)
          << " arousal=" << f2(is.arousal)
          << " wm_load=" << f1(is.wm_load)
          << " attn_focus=neuron" << (int)(is.attention_focus * brain_.som.n_neurons)
          << " mean_saliency=" << f2(is.mean_saliency)
          << " approach=" << (is.approach > 0.5f ? "YES" : "no");
        info("BrainState", o.str());
    }

    // ── Log WorkingMemory ──────────────────────────────────────────────────────
    void log_working_memory() {
        auto ctx = brain_.working_mem.context();
        int load = brain_.working_mem.size();
        int cap  = brain_.working_mem.get_base_capacity();
        info("WorkingMemory", "slots=" + std::to_string(load) + "/" + std::to_string(cap)
             + " context_norm=" + f2([&]{ float n=0; for(auto v:ctx) n+=v*v; return std::sqrt(n); }()));
    }

    // ── Log Attention/Novelty ──────────────────────────────────────────────────
    void log_attention(const std::string& label, const brain2::PerceiveResult& pr) {
        std::ostringstream o;
        o << label
          << " │ gate=" << (pr.attention_passed ? "OPEN " : "CLOSED")
          << " salience=" << f2(pr.salience)
          << " pred_error=" << f2(pr.prediction_error);
        // Classify novelty level
        if (pr.prediction_error > 0.5f)
            o << " [HIGH NOVELTY — strong attention spike]";
        else if (pr.prediction_error > 0.2f)
            o << " [moderate novelty]";
        else
            o << " [familiar — low novelty]";
        info("Attention", o.str());
    }

    // ── Log EpisodicMemory ─────────────────────────────────────────────────────
    void log_episodic(const brain2::PerceiveResult& pr) {
        std::ostringstream o;
        o << "stored=" << (pr.episodic_stored ? "YES" : "no")
          << " total_episodes=" << brain_.episodic.episode_count()
          << " surprise_threshold=" << f2(brain_.episodic_threshold());
        if (pr.episodic_stored)
            o << " [EPISODE COMMITTED — this event will be remembered]";
        info("EpisodicMemory", o.str());
    }

public:
    explicit CognitiveBridge(brain2::Brain& brain) : brain_(brain) {}

    // ── Main Entry: Full Cognitive Solve Cycle ─────────────────────────────────
    CognitiveSolveResult solve(const Problem& problem) {
        CognitiveSolveResult result{};
        result.problem_type = problem.type;
        solve_count_++;

        info("Bridge", "══════════════════════════════════════════════════════════");
        info("Bridge", "CYCLE #" + std::to_string(solve_count_)
             + " │ type='" + problem.type + "' │ data='" + problem.data_str + "'");
        info("Bridge", "══════════════════════════════════════════════════════════");

        // ── Pre-solve brain state ─────────────────────────────────────────────
        result.emotion_valence_before = brain_.emotion.valence;
        result.emotion_arousal_before = brain_.emotion.arousal;
        log_brain_state("PRE-SOLVE");

        // ── Step 1: PERCEIVE THE PROBLEM ─────────────────────────────────────
        info("Bridge", "┌─ STEP 1: PERCEIVE PROBLEM");
        auto problem_vec = encode_problem(problem);
        float novelty = som_novelty(problem_vec);  // before SOM updates
        result.som_novelty = novelty;

        auto pr1 = brain_.perceive(problem_vec);
        result.som_bmu          = pr1.bmu;
        result.prediction_error = pr1.prediction_error;
        result.attention_passed = pr1.attention_passed;
        result.wm_gate_open     = pr1.wm_passed;

        log_perceive("Perceived problem", pr1, novelty);
        log_attention("Attention", pr1);
        log_working_memory();
        log_episodic(pr1);
        brain_.tick();  // drain any daydream queued by the first perceive

        // ── Step 2: IMAGINATION — "What would happen if I solved this?" ───────
        info("Bridge", "├─ STEP 2: IMAGINATION (offline simulation from WM context)");
        auto sim = run_imagination(problem, true);
        result.imagination_ran       = true;
        result.imagination_coherence = sim.coherence;
        result.imagination_valence   = sim.valence;
        result.imagination_frames    = (int)sim.frames.size();

        // If imagination is highly coherent, it's a good sign the problem is solvable
        if (sim.coherence > 0.6f) {
            info("Bridge", "  → Imagination coherent (conf=" + f2(sim.coherence) + ") — proceeding with solve");
        } else {
            info("Bridge", "  → Imagination incoherent (conf=" + f2(sim.coherence) + ") — Brain uncertain, will try anyway");
        }

        // ── Step 3: SOLVE (Crisp Layer via UnifiedProposer) ──────────────────
        info("Bridge", "├─ STEP 3: UNIFIED PROPOSER SOLVE");
        bool solved = proposer_.solve(problem);
        result.solved = solved;

        // ── Step 4: EMOTION FEEDBACK ─────────────────────────────────────────
        info("Bridge", "├─ STEP 4: EMOTION FEEDBACK");
        std::string answer_label = solved ? ("solved_" + problem.type) : ("failed_" + problem.type);
        auto feedback_vec = encode_result(answer_label, solved);
        auto pr_feedback = brain_.perceive(feedback_vec);

        if (solved) {
            info("Emotion", "✓ Reward signal perceived → valence nudge ↑ │ "
                 "val=" + f2(pr_feedback.valence) + " aro=" + f2(pr_feedback.arousal));
        } else {
            warn("Emotion", "✗ Failure signal perceived → arousal ↑ │ "
                 "val=" + f2(pr_feedback.valence) + " aro=" + f2(pr_feedback.arousal));
        }

        // ── Step 5: WRITE TO FUZZY BINDING MEMORY ────────────────────────────
        info("Bridge", "├─ STEP 5: WRITE TO BINDING MEMORY");
        if (solved) {
            brain_.learn_from_crisp(problem.type, "solved_by", 1.0);
            info("BindingMemory", "Wrote: (" + problem.type + ", solved_by, 1.0)");
            for (const auto& [var, val] : problem.knowns) {
                brain_.learn_from_crisp(problem.type + "_" + var, "known_value", val);
                info("BindingMemory", "Wrote: (" + problem.type + "_" + var
                     + ", known_value, " + f2(val) + ")");
            }
        } else {
            info("BindingMemory", "Solve failed — no facts written");
        }

        // ── Step 6: RE-PERCEIVE THE ANSWER ────────────────────────────────────
        info("Bridge", "├─ STEP 6: RE-PERCEIVE ANSWER (SOM learns problem→solution topology)");
        auto answer_vec = encode_result(answer_label, solved);
        float ans_novelty = som_novelty(answer_vec);
        auto pr2 = brain_.perceive(answer_vec);

        log_perceive("Re-perceived answer", pr2, ans_novelty);
        result.episodic_stored = pr1.episodic_stored || pr2.episodic_stored;
        log_episodic(pr2);

        // ── Step 7: DAYDREAM (if InternalRouter queued it) ────────────────────
        info("Bridge", "├─ STEP 7: DAYDREAM / TICK");
        result.daydream_triggered = maybe_daydream(true);

        // ── Step 8: INNER SPEECH (think 3 steps from current WM context) ─────
        info("Bridge", "├─ STEP 8: INNER SPEECH (think from WM context after solve)");
        auto tr = run_think(3, true);
        result.inner_speech_words     = tr.words;
        result.inner_speech_coherence = tr.coherence;

        // ── Step 9: POST-SOLVE STATE SUMMARY ─────────────────────────────────
        result.emotion_valence_after = brain_.emotion.valence;
        result.emotion_arousal_after = brain_.emotion.arousal;
        log_brain_state("POST-SOLVE");
        log_working_memory();

        // Routing log entry for SleepEngine consolidation
        proposer_.print_routing_report();

        info("Bridge", "└─ CYCLE COMPLETE │ solved=" + std::string(solved ? "YES ✓" : "NO ✗")
             + " │ episodic=" + (result.episodic_stored ? "stored" : "no")
             + " │ daydream=" + (result.daydream_triggered ? "yes" : "no")
             + " │ imag_coh=" + f2(sim.coherence)
             + " │ inner_speech=[" + [&]{
                 std::string w;
                 for (const auto& wd : tr.words) { if (!w.empty()) w+=","; w+=wd; }
                 return w;
               }() + "]");
        info("Bridge", "══════════════════════════════════════════════════════════");

        return result;
    }

    // ── Batch Train ────────────────────────────────────────────────────────────
    struct TrainStats {
        int total = 0, solved = 0, episodic_stored = 0;
        int daydreams_triggered = 0;
        float avg_prediction_error = 0.f, avg_imagination_coherence = 0.f;
    };

    TrainStats train_batch(const std::vector<Problem>& problems) {
        TrainStats stats;
        float total_err = 0.f, total_coh = 0.f;
        for (const auto& p : problems) {
            auto r = solve(p);
            stats.total++;
            if (r.solved)               stats.solved++;
            if (r.episodic_stored)      stats.episodic_stored++;
            if (r.daydream_triggered)   stats.daydreams_triggered++;
            total_err += r.prediction_error;
            total_coh += r.imagination_coherence;
        }
        if (stats.total > 0) {
            stats.avg_prediction_error      = total_err / stats.total;
            stats.avg_imagination_coherence = total_coh / stats.total;
        }
        return stats;
    }

    // ── Persistence ────────────────────────────────────────────────────────────
    void save(const std::string& directory) {
        brain_.save_components(directory);
        proposer_.save_weights(directory + "/intuition.bin");
        info("Bridge", "All components saved → " + directory);
    }

    bool load(const std::string& directory) {
        try {
            brain_.load_components(
                directory + "/predictor.bin",
                directory + "/language.bin",
                directory + "/som.bin",
                directory + "/episodic.bin",
                directory + "/emotion.bin",
                directory + "/self.bin",
                directory + "/symbolic.bin",
                directory + "/binding.bin",
                directory + "/bg.bin",
                directory + "/procedures.bin",
                directory + "/hpred.bin",
                directory + "/decoder.bin"
            );
            proposer_.load_weights(directory + "/intuition.bin");
            info("Bridge", "All components loaded ← " + directory);
            return true;
        } catch (const std::exception& e) {
            warn("Bridge", "Load failed: " + std::string(e.what()) + " (starting fresh)");
            return false;
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────────
    brain2::Brain& get_brain()     { return brain_; }
    UnifiedProposer& get_proposer() { return proposer_; }

    std::pair<float, float> current_emotion() const {
        return {brain_.emotion.valence, brain_.emotion.arousal};
    }

    void print_state() const {
        auto is = brain_.build_internal_state();
        std::cout << "[Brain State] valence=" << std::fixed << std::setprecision(3) << is.valence
                  << " arousal=" << is.arousal
                  << " wm_load=" << is.wm_load
                  << " attn_focus=" << is.attention_focus
                  << " episodes=" << brain_.episodic.episode_count() << "\n";
    }

    void set_log_file(const std::string& path) {
        brain_log().set_log_file(path);
        info("Bridge", "Logging to file: " + path);
    }

    void set_log_level(BrainLog::Level l) {
        brain_log().set_level(l);
    }
};

} // namespace brain3
