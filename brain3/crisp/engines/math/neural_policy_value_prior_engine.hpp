#pragma once
/**
 * brain3/crisp/engines/math/neural_policy_value_prior_engine.hpp
 *
 * NEURAL POLICY & VALUE PRIOR — with ACTUAL LEARNING.
 *
 * v2 (audit fixes):
 *   - Policy P(a|s): per-tactic LINEAR MODELS over the 8-dim state
 *     embedding, trained by SGD on recorded (state, chosen, reward)
 *     outcomes. Replaces feature-dot + string-hash pseudo-logits.
 *   - Value V(s'): per-tactic affine heads regressed toward observed
 *     rewards. Replaces the hardcoded tactic-name table.
 *   - Replay buffer (cap 512) + train_pass() for batch consolidation;
 *     binary save/load persistence.
 *   - Untrained tactics fall back to v1 heuristic priors — graceful.
 */
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <algorithm>
#include <map>
#include <sstream>
#include <fstream>
#include <random>

#include "neural_guided_mcts_navigator.hpp"

namespace thebrain {
namespace neural_prior {

struct ProofStateEmbedding {
    std::string goal_id;
    int goal_ast_depth = 0;
    int hypothesis_count = 0;
    double algebraic_complexity_score = 0.0;
    double domain_embedding_weight = 1.0;
    std::vector<double> feature_vector;
};

struct CandidateActionScore {
    std::string tactic_name;
    std::string premise_id;
    double policy_prior_prob = 0.0;
    double value_estimate = 0.0;
    double combined_score = 0.0;
};

class NeuralPolicyValuePriorEngine {
public:
    explicit NeuralPolicyValuePriorEngine(double c_puct = 1.414)
        : c_puct_(c_puct) {}

    ProofStateEmbedding embed_proof_state(const std::string& goal_id,
                                          int ast_depth,
                                          int hyp_count,
                                          double complexity,
                                          const std::string& domain_str) {
        ProofStateEmbedding emb;
        emb.goal_id = goal_id;
        emb.goal_ast_depth = ast_depth;
        emb.hypothesis_count = hyp_count;
        emb.algebraic_complexity_score = complexity;

        double d_weight = 1.0;
        if (domain_str == "MATHEMATICS") d_weight = 1.2;
        else if (domain_str == "PHYSICS") d_weight = 1.1;
        emb.domain_embedding_weight = d_weight;

        auto& f = emb.feature_vector;
        f.resize(8);
        f[0] = std::tanh(ast_depth / 10.0);
        f[1] = std::tanh(hyp_count / 5.0);
        f[2] = std::tanh(complexity / 20.0);
        f[3] = d_weight / 2.0;
        f[4] = 1.0 / (1.0 + std::exp(-double(ast_depth)));
        f[5] = std::cos(ast_depth * 0.314);
        f[6] = std::sin(hyp_count * 0.628);
        f[7] = 0.5 * (f[0] + f[1]);
        return emb;
    }

    // LEARNED ranking: per-tactic linear models; heuristic fallback for
    // tactics that have never been trained.
    std::vector<CandidateActionScore> rank_candidate_actions(
        const ProofStateEmbedding& state_emb,
        const std::vector<std::pair<std::string, std::string>>& candidates,
        size_t top_k = 3) {
        std::vector<CandidateActionScore> scores;
        if (candidates.empty()) return scores;
        const auto& f = state_emb.feature_vector;

        std::vector<double> logits(candidates.size());
        std::vector<double> vals(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& tac = candidates[i].first;
            const TacticModel* tm = find_model(tac);
            if (tm) {
                logits[i] = dot(tm->pol_w, f) + tm->pol_b;
                vals[i]   = std::tanh(dot(tm->val_w, f) + tm->val_b);
            } else {
                logits[i] = heuristic_logit(tac, f);
                vals[i]   = heuristic_value(tac);
            }
        }

        double mx = *std::max_element(logits.begin(), logits.end());
        double Z = 0.;
        for (auto& l : logits) { l = std::exp(l - mx); Z += l; }

        for (size_t i = 0; i < candidates.size(); ++i) {
            CandidateActionScore c;
            c.tactic_name = candidates[i].first;
            c.premise_id  = candidates[i].second;
            c.policy_prior_prob = logits[i] / Z;
            c.value_estimate    = vals[i];
            c.combined_score    = c.value_estimate + c_puct_ * c.policy_prior_prob;
            scores.push_back(c);
        }
        std::sort(scores.begin(), scores.end(),
                  [](const CandidateActionScore& a, const CandidateActionScore& b){
                      return a.combined_score > b.combined_score;
                  });
        if (scores.size() > top_k) scores.resize(top_k);
        return scores;
    }

    // record: chosen tactic in this state produced `reward`
    void record_outcome(const ProofStateEmbedding& state_emb,
                        const std::string& tactic, double reward) {
        Sample smp;
        smp.features = state_emb.feature_vector;
        smp.tactic = tactic;
        smp.reward = reward;
        ensure_model(tactic);
        train_sample(models_[tactic], smp.features, reward);
        if (replay_.size() >= 512) replay_.erase(replay_.begin());
        replay_.push_back(smp);
    }

    // replay-batch consolidation (sleep hook); returns update count
    int train_pass(int epochs = 3, double lr = 0.05) {
        if (replay_.empty()) return 0;
        std::mt19937 g(77);
        int updates = 0;
        for (int ep = 0; ep < epochs; ++ep) {
            auto buf = replay_;
            std::shuffle(buf.begin(), buf.end(), g);
            for (auto& smp : buf) {
                train_sample(models_[smp.tactic], smp.features, smp.reward, lr);
                ++updates;
            }
        }
        return updates;
    }

    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t n = (uint32_t)models_.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& [tac, tm] : models_) {
            uint32_t len = (uint32_t)tac.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(len));
            f.write(tac.data(), len);
            auto wv = [&](const std::vector<double>& v){
                uint32_t sz = (uint32_t)v.size();
                f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
                f.write(reinterpret_cast<const char*>(v.data()), sz*sizeof(double));
            };
            wv(tm.pol_w); wv(tm.val_w);
            f.write(reinterpret_cast<const char*>(&tm.pol_b), sizeof(double));
            f.write(reinterpret_cast<const char*>(&tm.val_b), sizeof(double));
            f.write(reinterpret_cast<const char*>(&tm.n), sizeof(int));
        }
        return true;
    }
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t n = 0;
        if (!f.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
        models_.clear();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t len = 0;
            if (!f.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
            std::string tac(len, ' ');
            if (!f.read(tac.data(), len)) return false;
            TacticModel& tm = models_[tac];
            auto rv = [&](std::vector<double>& v){
                uint32_t sz = 0;
                f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
                v.resize(sz);
                f.read(reinterpret_cast<char*>(v.data()), sz*sizeof(double));
            };
            rv(tm.pol_w); rv(tm.val_w);
            f.read(reinterpret_cast<char*>(&tm.pol_b), sizeof(double));
            f.read(reinterpret_cast<char*>(&tm.val_b), sizeof(double));
            f.read(reinterpret_cast<char*>(&tm.n), sizeof(int));
        }
        return true;
    }

private:
    static constexpr size_t FD = 8;

    struct TacticModel {
        std::vector<double> pol_w = std::vector<double>(FD, 0.0);
        double pol_b = 0.0;
        std::vector<double> val_w = std::vector<double>(FD, 0.0);
        double val_b = 0.0;
        int n = 0;
    };
    struct Sample { std::vector<double> features; std::string tactic; double reward; };

    std::map<std::string, TacticModel> models_;
    std::vector<Sample> replay_;
    double c_puct_ = 1.414;

    static double dot(const std::vector<double>& w, const std::vector<double>& f) {
        double s = 0.;
        for (size_t i = 0; i < FD && i < f.size(); ++i) s += w[i] * f[i];
        return s;
    }
    TacticModel* find_model(const std::string& tac) {
        auto it = models_.find(tac);
        return it == models_.end() ? nullptr : &it->second;
    }
    const TacticModel* find_model(const std::string& tac) const {
        auto it = models_.find(tac);
        return it == models_.end() ? nullptr : &it->second;
    }
    void ensure_model(const std::string& tac) {
        if (!models_.count(tac)) models_[tac] = TacticModel{};
    }

    void train_sample(TacticModel& tm, const std::vector<double>& f,
                      const std::string&, double reward, double lr) {
        double vp = dot(tm.val_w, f) + tm.val_b;
        double vpred = std::tanh(vp);
        double vg = (vpred - reward) * (1.0 - vpred * vpred);
        // signed policy target: success pushes UP, failure pushes DOWN
        const double t = (reward > 0 ? 1.0 : -1.0) * (0.5 + 0.5 * std::fabs(reward));
        for (int j = 0; j < FD && j < (int)f.size(); ++j) {
            tm.val_w[j] -= lr * vg * f[j];
            tm.pol_w[j] += lr * t * f[j];
        }
        tm.val_b -= lr * vg;
        tm.pol_b += lr * t;
    }
    void train_sample(TacticModel& tm, const std::vector<double>& f, double reward) {
        train_sample(tm, f, reward, 0.05);
    }
    void train_sample(TacticModel& tm, const std::vector<double>& f,
                      double reward, double lr) {
        train_sample(tm, f, std::string(), reward, lr);
    }

    static double heuristic_logit(const std::string& tactic,
                                  const std::vector<double>& f) {
        double hash_t = (double)(std::hash<std::string>{}(tactic) % 100) / 100.0;
        double logit = 0.;
        for (size_t i = 0; i < f.size(); ++i)
            logit += f[i] * (0.5 + 0.1 * i);
        return logit + hash_t * 0.2;
    }
    static double heuristic_value(const std::string& tactic) {
        if (tactic == "exact" || tactic == "ring" || tactic == "linarith")
            return std::tanh(0.85);
        if (tactic == "apply" || tactic == "have") return std::tanh(0.65);
        return std::tanh(0.5);
    }
};

} // namespace neural_prior
} // namespace thebrain
