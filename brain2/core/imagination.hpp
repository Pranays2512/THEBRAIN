#pragma once
/*
 * imagination.hpp — Imagination (Offline Simulation), Component 5 of Brain v2
 *
 * Runs the Predictor in offline mode — simulates sequences of activations
 * that haven't happened yet. Same prediction engine as online processing,
 * but disconnected from real input and no weight updates.
 *
 * "What if?" — brain feeds hypothetical start state → simulates N steps forward
 * → evaluates outcome (good/bad) → informs decisions without real-world cost.
 *
 * Dreams = imagination running during rest with random or memory-seeded starts.
 *
 * Output: sequence of predicted activation vectors + quality score
 * Quality = mean similarity between consecutive frames (coherence measure)
 */

#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include "predictor.hpp"

namespace brain2 {

struct Simulation {
    std::vector<std::vector<float>> frames;  // predicted activation sequence
    float coherence;  // mean cosine similarity between consecutive frames
    float valence;    // assigned after evaluation (positive/negative outcome)
    bool  completed;  // did simulation run to completion
};

class Imagination {
public:
    int n_dims;
    int max_steps;   // max steps per simulation

private:
    Predictor*                  predictor_;  // shared — does NOT own
    std::unique_ptr<std::mutex> mtx_;

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

    float sequence_coherence(const std::vector<std::vector<float>>& frames) const {
        if (frames.size() < 2) return 1.f;
        float sum = 0.f;
        for (size_t i = 1; i < frames.size(); i++)
            sum += cosine(frames[i-1], frames[i]);
        return sum / (frames.size() - 1);
    }

public:
    Imagination() : n_dims(0), max_steps(20), predictor_(nullptr),
                    mtx_(std::make_unique<std::mutex>()) {}

    // predictor: non-owning pointer — Predictor must outlive Imagination
    Imagination(Predictor* predictor, int max_steps = 20)
        : n_dims(predictor ? predictor->input_dim : 0),
          max_steps(max_steps), predictor_(predictor),
          mtx_(std::make_unique<std::mutex>()) {}

    Imagination(Imagination&&)            = default;
    Imagination& operator=(Imagination&&) = default;
    Imagination(const Imagination&)       = delete;
    Imagination& operator=(const Imagination&) = delete;

    // Simulate N steps forward from a start state
    // No weight updates — offline mode
    Simulation simulate(const std::vector<float>& start_state,
                        int steps = -1) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (!predictor_) return {{{start_state}}, 0.f, 0.f, false};

        int n = (steps > 0) ? std::min(steps, max_steps) : max_steps;

        bool was_offline = predictor_->is_offline();
        predictor_->set_offline(true);
        predictor_->reset();

        Simulation sim;
        sim.frames.push_back(start_state);
        sim.valence   = 0.f;
        sim.completed = false;

        std::vector<float> current = start_state;
        for (int i = 0; i < n; i++) {
            auto next = predictor_->step(current);
            sim.frames.push_back(next);
            current = next;
        }
        sim.completed = true;
        sim.coherence = sequence_coherence(sim.frames);

        predictor_->set_offline(was_offline);
        return sim;
    }

    // Dream: simulate from random or episodic-seeded start
    // Used during rest to consolidate and explore
    std::vector<Simulation> dream(int n_dreams, int steps_per_dream,
                                  const std::vector<std::vector<float>>& seeds,
                                  unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::vector<Simulation> results;
        results.reserve(n_dreams);

        for (int d = 0; d < n_dreams; d++) {
            std::vector<float> start;
            if (!seeds.empty()) {
                // Pick random seed from provided episodic memories
                std::uniform_int_distribution<int> pick(0, (int)seeds.size()-1);
                start = seeds[pick(rng)];
                // Add small noise to explore variations
                std::normal_distribution<float> noise(0.f, 0.05f);
                for (auto& x : start) x += noise(rng);
            } else {
                // Random start
                start.resize(n_dims);
                std::normal_distribution<float> nd(0.f, 0.3f);
                for (auto& x : start) x = nd(rng);
            }
            results.push_back(simulate(start, steps_per_dream));
        }
        return results;
    }

    // Evaluate simulation outcome against a goal state
    // Returns valence: +1 = goal reached, -1 = diverged, 0 = neutral
    float evaluate(const Simulation& sim,
                   const std::vector<float>& goal_state,
                   float threshold = 0.8f) const {
        if (sim.frames.empty()) return 0.f;
        // Check if any frame is close to goal
        float max_sim = 0.f;
        for (const auto& f : sim.frames)
            max_sim = std::max(max_sim, cosine(f, goal_state));
        if (max_sim >= threshold) return 1.f;
        if (sim.coherence < 0.3f) return -0.5f;  // incoherent = bad
        return (max_sim - 0.5f) * 2.f;  // scaled [-1, 1]
    }

    // Extract all unique activation frames from a batch of simulations
    // Useful for seeding working memory or training
    std::vector<std::vector<float>> extract_frames(
            const std::vector<Simulation>& sims,
            float min_coherence = 0.4f) const {
        std::vector<std::vector<float>> out;
        for (const auto& s : sims) {
            if (s.coherence < min_coherence) continue;
            for (const auto& f : s.frames)
                out.push_back(f);
        }
        return out;
    }

    bool has_predictor() const noexcept { return predictor_ != nullptr; }
};

} // namespace brain2
