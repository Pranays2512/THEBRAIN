#pragma once
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

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <memory>
#include <fstream>
#include <stdexcept>

namespace brain2 {

struct AttentionResult {
    bool   passed;      // did this activation pass the gate?
    float  score;       // attention score for this input
    float  threshold;   // threshold that was applied
    int    focus_bmu;   // BMU with highest saliency after update
};

class Attention {
public:
    int   n_neurons;
    float decay_rate;       // saliency decay per tick
    float base_threshold;   // minimum gate threshold

private:
    std::vector<float>       saliency_;     // per-neuron saliency map
    std::vector<float>       top_down_bias_;// goal-directed bias
    float                    current_threshold_;
    std::unique_ptr<std::mutex> mtx_;

    static float clamp(float v, float lo, float hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }

public:
    Attention() : n_neurons(0), decay_rate(0.1f), base_threshold(0.3f),
                  current_threshold_(0.3f),
                  mtx_(std::make_unique<std::mutex>()) {}

    Attention(int n_neurons, float decay_rate = 0.1f, float base_threshold = 0.3f)
        : n_neurons(n_neurons), decay_rate(decay_rate),
          base_threshold(base_threshold),
          current_threshold_(base_threshold),
          saliency_(n_neurons, 0.f),
          top_down_bias_(n_neurons, 0.f),
          mtx_(std::make_unique<std::mutex>()) {}

    Attention(Attention&&)            = default;
    Attention& operator=(Attention&&) = default;
    Attention(const Attention&)       = delete;
    Attention& operator=(const Attention&) = delete;

    // Core gate: decide if activation_map passes attention, update saliency
    // activation_map: normalized neuron activations from SOM (0–1, length n_neurons)
    // novelty: prediction error [0, 1] — drives saliency update
    // arousal_modulator: from Emotion::attention_modulator() [0.5, 1.0]
    AttentionResult gate(const std::vector<float>& activation_map,
                         float novelty,
                         float arousal_modulator = 0.75f) {
        std::lock_guard<std::mutex> lock(*mtx_);

        int n = std::min((int)activation_map.size(), n_neurons);

        // Update saliency: decay + novelty injection at active neurons
        int focus_bmu = 0;
        float max_sal = -1.f;
        for (int i = 0; i < n_neurons; i++) {
            float act = (i < (int)activation_map.size()) ? activation_map[i] : 0.f;
            saliency_[i] = saliency_[i] * (1.f - decay_rate)
                         + novelty * act
                         + top_down_bias_[i] * 0.1f;
            saliency_[i] = clamp(saliency_[i], 0.f, 1.f);
            if (saliency_[i] > max_sal) {
                max_sal = saliency_[i]; focus_bmu = i;
            }
        }

        // Compute attention score = dot product of saliency and activation
        float score = 0.f;
        float sal_norm = 0.f;
        for (int i = 0; i < n_neurons; i++) {
            float act = (i < (int)activation_map.size()) ? activation_map[i] : 0.f;
            score    += saliency_[i] * act;
            sal_norm += saliency_[i];
        }
        if (sal_norm > 1e-8f) score /= sal_norm;

        // Threshold modulated by emotional arousal
        current_threshold_ = base_threshold * arousal_modulator * 2.f;
        current_threshold_ = clamp(current_threshold_, 0.05f, 0.95f);

        bool passed = score >= current_threshold_;
        return {passed, score, current_threshold_, focus_bmu};
    }

    // Set top-down bias (goal-directed attention)
    // Caller provides a concept vector; we project to neuron space via simple mean
    void set_top_down(const std::vector<float>& neuron_weights) {
        std::lock_guard<std::mutex> lock(*mtx_);
        int n = std::min((int)neuron_weights.size(), n_neurons);
        for (int i = 0; i < n; i++)
            top_down_bias_[i] = clamp(neuron_weights[i], 0.f, 1.f);
    }

    void clear_top_down() {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::fill(top_down_bias_.begin(), top_down_bias_.end(), 0.f);
    }

    // Decay saliency each time step
    void tick() {
        std::lock_guard<std::mutex> lock(*mtx_);
        for (auto& s : saliency_)
            s *= (1.f - decay_rate);
    }

    // Return current saliency map copy
    std::vector<float> saliency_map() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return saliency_;
    }

    // Focus neuron = highest saliency
    int focus_neuron() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return (int)(std::max_element(saliency_.begin(), saliency_.end()) - saliency_.begin());
    }

    // Mean saliency — overall alertness level
    float mean_saliency() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (saliency_.empty()) return 0.f;
        float s = 0.f;
        for (auto v : saliency_) s += v;
        return s / (float)saliency_.size();
    }

    float threshold() const noexcept { return current_threshold_; }

    void reset() {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::fill(saliency_.begin(),      saliency_.end(),      0.f);
        std::fill(top_down_bias_.begin(), top_down_bias_.end(), 0.f);
        current_threshold_ = base_threshold;
    }

    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Attention::save: cannot open " + path);
        f.write((const char*)&n_neurons,          sizeof(int));
        f.write((const char*)&decay_rate,         sizeof(float));
        f.write((const char*)&base_threshold,     sizeof(float));
        f.write((const char*)&current_threshold_, sizeof(float));
        f.write((const char*)saliency_.data(),
                (std::streamsize)(n_neurons * sizeof(float)));
        f.write((const char*)top_down_bias_.data(),
                (std::streamsize)(n_neurons * sizeof(float)));
    }

    static Attention load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Attention::load: cannot open " + path);
        Attention a;
        f.read((char*)&a.n_neurons,          sizeof(int));
        f.read((char*)&a.decay_rate,         sizeof(float));
        f.read((char*)&a.base_threshold,     sizeof(float));
        f.read((char*)&a.current_threshold_, sizeof(float));
        a.saliency_.resize(a.n_neurons);
        a.top_down_bias_.resize(a.n_neurons);
        f.read((char*)a.saliency_.data(),
               (std::streamsize)(a.n_neurons * sizeof(float)));
        f.read((char*)a.top_down_bias_.data(),
               (std::streamsize)(a.n_neurons * sizeof(float)));
        a.mtx_ = std::make_unique<std::mutex>();
        return a;
    }
};

} // namespace brain2
