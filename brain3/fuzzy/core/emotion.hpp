#pragma once
/*
 * emotion.hpp — Emotion System, Component 6 of Brain v2
 *
 * Valence   [-1, +1]: negative (fear/pain) → positive (reward/pleasure)
 * Arousal   [ 0,  1]: calm (sleep) → excited (panic/joy)
 *
 * Emotion is NOT a separate module — it's a global modulation signal:
 *   - High arousal → boosts attention threshold (more selective)
 *   - Positive valence + high arousal → exploration mode
 *   - Negative valence + high arousal → avoidance mode
 *   - Emotion modulates learning rate (surprise * arousal amplifier)
 *   - Salience for working memory gating = arousal * abs(valence)
 *
 * Learning:
 *   Emotion state drifts toward "emotional input" (valenced prediction error).
 *   decay_rate moves back toward neutral when no signal.
 *   Strong events leave emotional traces that decay slowly.
 *
 * Trigger sources:
 *   - Prediction error (surprise = arousal increase)
 *   - Goal proximity (positive valence)
 *   - Threat signals (negative valence, high arousal)
 *   - Internal needs (hunger/fatigue-like drives — abstracted as need_level)
 */

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <memory>
#include <fstream>
#include <stdexcept>

namespace brain2 {

struct EmotionState {
    float valence;  // [-1, +1]
    float arousal;  // [ 0,  1]
};

struct EmotionEvent {
    float valence_delta;  // how much to push valence
    float arousal_delta;  // how much to push arousal
    float intensity;      // [0, 1] — scales both deltas
};

class Emotion {
public:
    float valence;    // current emotional state
    float arousal;
    float decay_rate; // how fast emotion returns to neutral (per tick)

private:
    float peak_valence_;   // recent peak — emotional memory trace
    float peak_arousal_;
    float peak_decay_;     // peak decays slower than current state

    std::unique_ptr<std::mutex> mtx_;

    static float clamp(float v, float lo, float hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    static float lerp(float a, float b, float t) noexcept {
        return a + t * (b - a);
    }

public:
    Emotion() : valence(0.f), arousal(0.f), decay_rate(0.05f),
                peak_valence_(0.f), peak_arousal_(0.f), peak_decay_(0.01f),
                mtx_(std::make_unique<std::mutex>()) {}

    Emotion(float decay_rate, float peak_decay = 0.01f)
        : valence(0.f), arousal(0.f), decay_rate(decay_rate),
          peak_valence_(0.f), peak_arousal_(0.f), peak_decay_(peak_decay),
          mtx_(std::make_unique<std::mutex>()) {}

    Emotion(Emotion&&)            = default;
    Emotion& operator=(Emotion&&) = default;
    Emotion(const Emotion&)       = delete;
    Emotion& operator=(const Emotion&) = delete;

    // Trigger an emotional event — updates valence and arousal
    void trigger(const EmotionEvent& event) {
        std::lock_guard<std::mutex> lock(*mtx_);
        float v_delta = event.valence_delta * event.intensity;
        float a_delta = event.arousal_delta * event.intensity;

        valence = clamp(valence + v_delta, -1.f, 1.f);
        arousal = clamp(arousal + a_delta,  0.f, 1.f);

        // Update peaks if exceeded
        if (std::abs(valence) > std::abs(peak_valence_))
            peak_valence_ = valence;
        if (arousal > peak_arousal_)
            peak_arousal_ = arousal;
    }

    // Convenience: trigger from raw prediction error (surprise → arousal + mild negative)
    // high error = surprised = aroused; surprise itself is neutral but alerting
    void from_prediction_error(float error) {
        float normed = clamp(error, 0.f, 1.f);
        // Surprise → arousal spike, mild negative valence (unknown = slightly bad)
        trigger({-0.1f * normed, normed * 0.5f, normed});
    }

    // Trigger from explicit reward signal
    void from_reward(float reward) {
        // reward ∈ [-1, 1] → both valence and arousal shift
        float intensity = std::abs(reward);
        float v_dir = reward > 0.f ? 1.f : -1.f;
        trigger({v_dir * 0.8f, 0.3f, intensity});
    }

    // Decay toward neutral (call every time step)
    void tick() {
        std::lock_guard<std::mutex> lock(*mtx_);
        // Current state → neutral
        valence = lerp(valence, 0.f, decay_rate);
        arousal = lerp(arousal, 0.f, decay_rate);
        // Peaks decay slower
        peak_valence_ = lerp(peak_valence_, 0.f, peak_decay_);
        peak_arousal_ = lerp(peak_arousal_, 0.f, peak_decay_);
    }

    // Salience for working memory: how important is current emotional state
    float salience() const noexcept {
        return arousal * (0.5f + 0.5f * std::abs(valence));
    }

    // When false, lr_modulator returns the neutral 1.0 regardless of arousal —
    // ablates emotion's salience-weighted learning so its effect is measurable.
    bool modulation_enabled = true;

    // Learning rate modulator: aroused + surprised = learn more
    float lr_modulator() const noexcept {
        // Neutral state = 1.0x, max arousal = 2.0x
        return modulation_enabled ? (1.f + arousal) : 1.f;
    }

    // Attention threshold modulator: high arousal = more selective attention
    float attention_modulator() const noexcept {
        return 0.5f + 0.5f * arousal;
    }

    // Is brain in approach mode? (want to seek/explore)
    bool approach_mode() const noexcept {
        return valence > 0.1f && arousal > 0.2f;
    }

    // Is brain in avoidance mode? (want to flee/avoid)
    bool avoidance_mode() const noexcept {
        return valence < -0.1f && arousal > 0.2f;
    }

    // Emotional inertia: how resistant to change (high peak = hard to shift)
    float inertia() const noexcept {
        return 0.3f * std::abs(peak_valence_) + 0.3f * peak_arousal_;
    }

    // Snapshot for serialization
    EmotionState state() const noexcept {
        return {valence, arousal};
    }

    float peak_valence() const noexcept { return peak_valence_; }
    float peak_arousal() const noexcept { return peak_arousal_; }

    void reset() {
        std::lock_guard<std::mutex> lock(*mtx_);
        valence = 0.f; arousal = 0.f;
        peak_valence_ = 0.f; peak_arousal_ = 0.f;
    }

    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Emotion::save: cannot open " + path);
        f.write((const char*)&valence,       sizeof(float));
        f.write((const char*)&arousal,       sizeof(float));
        f.write((const char*)&decay_rate,    sizeof(float));
        f.write((const char*)&peak_valence_, sizeof(float));
        f.write((const char*)&peak_arousal_, sizeof(float));
        f.write((const char*)&peak_decay_,   sizeof(float));
    }

    static Emotion load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Emotion::load: cannot open " + path);
        Emotion e;
        f.read((char*)&e.valence,       sizeof(float));
        f.read((char*)&e.arousal,       sizeof(float));
        f.read((char*)&e.decay_rate,    sizeof(float));
        f.read((char*)&e.peak_valence_, sizeof(float));
        f.read((char*)&e.peak_arousal_, sizeof(float));
        f.read((char*)&e.peak_decay_,   sizeof(float));
        e.mtx_ = std::make_unique<std::mutex>();
        return e;
    }
};

} // namespace brain2
