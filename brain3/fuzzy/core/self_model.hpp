#pragma once
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

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <mutex>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <random>

namespace brain2 {

// Snapshot of brain's internal state — input to self-model
struct InternalState {
    float valence;          // emotion valence [-1, 1]
    float arousal;          // emotion arousal [0, 1]
    float salience;         // current attention salience [0, 1]
    float pred_error;       // recent prediction error [0, 1]
    float wm_load;          // working memory utilization [0, 1]
    float attention_focus;  // normalized focus neuron index [0, 1]
    float mean_saliency;    // overall alertness [0, 1]
    float approach;         // approach mode indicator [0, 1]
    float avoidance;        // avoidance mode indicator [0, 1]
    float arousal_trend;    // recent arousal change [-1, 1]
};

static constexpr int SELF_STATE_DIM = 10;

inline std::vector<float> state_to_vec(const InternalState& s) {
    return {
        s.valence, s.arousal, s.salience, s.pred_error,
        s.wm_load, s.attention_focus, s.mean_saliency,
        s.approach, s.avoidance, s.arousal_trend
    };
}

// Mini-SOM for self-concept clustering (no dependency on SOM class)
class SelfSOM {
public:
    int n_neurons;
    int n_dims;
    float lr;
    float radius;
    std::vector<std::vector<float>> weights;

    SelfSOM() : n_neurons(0), n_dims(0), lr(0.1f), radius(3.f) {}

    SelfSOM(int n_neurons, int n_dims, float lr = 0.1f, unsigned seed = 42)
        : n_neurons(n_neurons), n_dims(n_dims), lr(lr), radius(float(n_neurons) * 0.3f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        weights.resize(n_neurons, std::vector<float>(n_dims));
        for (auto& w : weights)
            for (auto& x : w) x = dist(rng);
    }

    int find_bmu(const std::vector<float>& v) const {
        int bmu = 0; float best = 1e30f;
        for (int i = 0; i < n_neurons; i++) {
            float d = 0.f;
            for (int j = 0; j < n_dims; j++) {
                float diff = v[j] - weights[i][j];
                d += diff * diff;
            }
            if (d < best) { best = d; bmu = i; }
        }
        return bmu;
    }

    void update(const std::vector<float>& v, int bmu) {
        for (int i = 0; i < n_neurons; i++) {
            float dist = float(std::abs(i - bmu));
            float h = std::exp(-dist*dist / (2.f * radius * radius));
            for (int j = 0; j < n_dims; j++)
                weights[i][j] += lr * h * (v[j] - weights[i][j]);
        }
        // Slowly shrink radius
        radius = std::max(0.5f, radius * 0.9999f);
        lr     = std::max(0.001f, lr * 0.9999f);
    }
};

class SelfModel {
public:
    int n_self_neurons;

private:
    SelfSOM                   self_som_;
    std::vector<float>        identity_;     // running mean of all states
    std::deque<float>         error_history_;// recent prediction errors
    std::deque<float>         arousal_history_;
    int                       obs_count_;
    float                     prev_arousal_;
    std::unique_ptr<std::mutex> mtx_;

    static float clamp(float v, float lo, float hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    static float cosine(const std::vector<float>& a,
                        const std::vector<float>& b) noexcept {
        float dot = 0.f, na = 0.f, nb = 0.f;
        for (size_t i = 0; i < a.size() && i < b.size(); i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

public:
    SelfModel() : n_self_neurons(0), obs_count_(0), prev_arousal_(0.f),
                  mtx_(std::make_unique<std::mutex>()) {}

    SelfModel(int n_self_neurons, unsigned seed = 42)
        : n_self_neurons(n_self_neurons),
          self_som_(n_self_neurons, SELF_STATE_DIM, 0.1f, seed),
          identity_(SELF_STATE_DIM, 0.f),
          obs_count_(0), prev_arousal_(0.f),
          mtx_(std::make_unique<std::mutex>()) {}

    SelfModel(SelfModel&&)            = default;
    SelfModel& operator=(SelfModel&&) = default;
    SelfModel(const SelfModel&)       = delete;
    SelfModel& operator=(const SelfModel&) = delete;

    // Observe current brain state — core update
    void observe(const InternalState& state) {
        std::lock_guard<std::mutex> lock(*mtx_);

        error_history_.push_back(state.pred_error);
        if ((int)error_history_.size() > 20) error_history_.pop_front();

        arousal_history_.push_back(state.arousal);
        if ((int)arousal_history_.size() > 10) arousal_history_.pop_front();

        auto vec = state_to_vec(state);
        // Update self-concept SOM
        int bmu = self_som_.find_bmu(vec);
        self_som_.update(vec, bmu);

        // Update running identity (mean of all observations)
        obs_count_++;
        float alpha = 1.f / float(obs_count_);
        for (int i = 0; i < SELF_STATE_DIM; i++)
            identity_[i] = identity_[i] * (1.f - alpha) + vec[i] * alpha;

        prev_arousal_ = state.arousal;
    }

    // Current self-concept neuron
    int current_concept(const InternalState& state) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto vec = state_to_vec(state);
        return self_som_.find_bmu(vec);
    }

    // How far is current state from typical identity? (0=normal, 1=anomalous)
    float drift(const InternalState& state) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (obs_count_ < 5) return 0.f;
        auto vec = state_to_vec(state);
        float sim = cosine(vec, identity_);
        return clamp(1.f - sim, 0.f, 1.f);
    }

    // Mean recent prediction error
    float mean_recent_error() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (error_history_.empty()) return 0.f;
        float s = 0.f;
        for (auto e : error_history_) s += e;
        return s / (float)error_history_.size();
    }

    // Arousal trend: positive = increasing, negative = decreasing
    float arousal_trend() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        int n = (int)arousal_history_.size();
        if (n < 2) return 0.f;
        float delta = 0.f;
        for (int i = 1; i < n; i++)
            delta += arousal_history_[i] - arousal_history_[i-1];
        return clamp(delta / (float)(n-1), -1.f, 1.f);
    }

    // Identity vector (what "I" look like on average)
    std::vector<float> identity() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return identity_;
    }

    // Self-concept weights for neuron i
    std::vector<float> concept_weights(int i) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (i < 0 || i >= n_self_neurons) return {};
        return self_som_.weights[i];
    }

    int obs_count() const noexcept { return obs_count_; }

    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SelfModel::save: cannot open " + path);
        f.write((const char*)&n_self_neurons, sizeof(int));
        f.write((const char*)&obs_count_,     sizeof(int));
        f.write((const char*)&prev_arousal_,  sizeof(float));
        // SOM weights
        for (const auto& w : self_som_.weights)
            f.write((const char*)w.data(),
                    (std::streamsize)(SELF_STATE_DIM * sizeof(float)));
        f.write((const char*)&self_som_.lr,     sizeof(float));
        f.write((const char*)&self_som_.radius, sizeof(float));
        // Identity
        f.write((const char*)identity_.data(),
                (std::streamsize)(SELF_STATE_DIM * sizeof(float)));
    }

    static SelfModel load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SelfModel::load: cannot open " + path);
        SelfModel sm;
        f.read((char*)&sm.n_self_neurons, sizeof(int));
        f.read((char*)&sm.obs_count_,     sizeof(int));
        f.read((char*)&sm.prev_arousal_,  sizeof(float));
        sm.self_som_ = SelfSOM(sm.n_self_neurons, SELF_STATE_DIM);
        for (auto& w : sm.self_som_.weights)
            f.read((char*)w.data(),
                   (std::streamsize)(SELF_STATE_DIM * sizeof(float)));
        f.read((char*)&sm.self_som_.lr,     sizeof(float));
        f.read((char*)&sm.self_som_.radius, sizeof(float));
        sm.identity_.resize(SELF_STATE_DIM);
        f.read((char*)sm.identity_.data(),
               (std::streamsize)(SELF_STATE_DIM * sizeof(float)));
        sm.mtx_ = std::make_unique<std::mutex>();
        return sm;
    }
};

} // namespace brain2
