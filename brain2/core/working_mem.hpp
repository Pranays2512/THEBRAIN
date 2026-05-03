#pragma once
/*
 * working_mem.hpp — Working Memory (Prefrontal Cortex), Component 3 of Brain v2
 *
 * Holds 7±2 currently active concept vectors.
 * Maintains thread of thought across time steps.
 *
 * Gating policy:
 *   - New item enters if it passes relevance threshold OR slots available
 *   - When full: least relevant item (lowest dot product with context) dropped
 *   - Emotional salience (external signal) can force retention
 *   - Items decay each tick — unused thoughts fade
 *
 * Context vector = weighted average of all current slots.
 * This is what the Predictor and Language components read.
 */

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <string>
#include <limits>

namespace brain2 {

struct WMSlot {
    std::vector<float> vec;
    float salience;   // emotional weight — high salience = resist eviction
    float activation; // decays each tick
    int   age;        // ticks since inserted
};

class WorkingMemory {
public:
    int   n_dims;
    int   capacity;   // max slots (default 7)
    float decay_rate; // activation decay per tick (default 0.95)

private:
    std::vector<WMSlot>         slots_;
    std::unique_ptr<std::mutex> mtx_;

    // Dot product relevance between two vectors
    static float dot(const std::vector<float>& a,
                     const std::vector<float>& b) noexcept {
        float s = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    // L2 norm
    static float norm(const std::vector<float>& v) noexcept {
        float s = 0.f;
        for (auto x : v) s += x * x;
        return std::sqrt(s);
    }

    // Cosine similarity
    static float cosine(const std::vector<float>& a,
                        const std::vector<float>& b) noexcept {
        float na = norm(a), nb = norm(b);
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot(a, b) / (na * nb);
    }

    // Index of least important slot (lowest activation * salience)
    int least_important() const noexcept {
        int idx = 0;
        float min_val = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)slots_.size(); i++) {
            float val = slots_[i].activation * (1.f + slots_[i].salience);
            if (val < min_val) { min_val = val; idx = i; }
        }
        return idx;
    }

public:
    WorkingMemory() : n_dims(0), capacity(7), decay_rate(0.95f),
                      mtx_(std::make_unique<std::mutex>()) {}

    WorkingMemory(int n_dims, int capacity = 7, float decay_rate = 0.95f)
        : n_dims(n_dims), capacity(capacity), decay_rate(decay_rate),
          mtx_(std::make_unique<std::mutex>())
    {
        slots_.reserve(capacity);
    }

    WorkingMemory(WorkingMemory&&)            = default;
    WorkingMemory& operator=(WorkingMemory&&) = default;
    WorkingMemory(const WorkingMemory&)       = delete;
    WorkingMemory& operator=(const WorkingMemory&) = delete;

    // Try to insert new activation into working memory
    // salience: emotional weight [0,1] — high = resist eviction
    // Returns true if inserted
    bool gate(const std::vector<float>& activation, float salience = 0.f) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if ((int)activation.size() != n_dims) return false;

        // Check if similar item already present — update it instead
        for (auto& s : slots_) {
            if (cosine(s.vec, activation) > 0.9f) {
                s.activation = std::min(1.f, s.activation + 0.3f);
                s.salience   = std::max(s.salience, salience);
                s.age        = 0;
                return true;
            }
        }

        WMSlot slot;
        slot.vec        = activation;
        slot.salience   = salience;
        slot.activation = 1.f;
        slot.age        = 0;

        if ((int)slots_.size() < capacity) {
            slots_.push_back(std::move(slot));
        } else {
            // Evict least important
            int idx = least_important();
            slots_[idx] = std::move(slot);
        }
        return true;
    }

    // Decay all slots — call each tick
    void tick() {
        std::lock_guard<std::mutex> lock(*mtx_);
        for (auto& s : slots_) {
            s.activation *= decay_rate;
            s.age++;
        }
        // Remove fully decayed slots
        slots_.erase(
            std::remove_if(slots_.begin(), slots_.end(),
                [](const WMSlot& s){ return s.activation < 0.01f; }),
            slots_.end());
    }

    // Context vector: weighted mean of all slots by activation
    std::vector<float> context() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (slots_.empty()) return std::vector<float>(n_dims, 0.f);
        std::vector<float> ctx(n_dims, 0.f);
        float total = 0.f;
        for (const auto& s : slots_) {
            float w = s.activation;
            for (int i = 0; i < n_dims; i++) ctx[i] += w * s.vec[i];
            total += w;
        }
        if (total > 0.f)
            for (auto& x : ctx) x /= total;
        return ctx;
    }

    // Most active slot vector
    std::vector<float> most_active() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (slots_.empty()) return std::vector<float>(n_dims, 0.f);
        const WMSlot* best = &slots_[0];
        for (const auto& s : slots_)
            if (s.activation > best->activation) best = &s;
        return best->vec;
    }

    // Boost salience of slot most similar to given vector (emotion signal)
    void boost_salience(const std::vector<float>& v, float amount) {
        std::lock_guard<std::mutex> lock(*mtx_);
        float best_sim = -1.f;
        int   best_idx = -1;
        for (int i = 0; i < (int)slots_.size(); i++) {
            float sim = cosine(slots_[i].vec, v);
            if (sim > best_sim) { best_sim = sim; best_idx = i; }
        }
        if (best_idx >= 0)
            slots_[best_idx].salience = std::min(1.f,
                slots_[best_idx].salience + amount);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(*mtx_);
        slots_.clear();
    }

    int   size()  const noexcept { return (int)slots_.size(); }
    bool  empty() const noexcept { return slots_.empty(); }

    // Return all slot activations (for inspection/testing)
    std::vector<float> activations() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<float> a;
        a.reserve(slots_.size());
        for (const auto& s : slots_) a.push_back(s.activation);
        return a;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("WorkingMemory::save: cannot open " + path);
        f.write((const char*)&n_dims,      sizeof(int));
        f.write((const char*)&capacity,    sizeof(int));
        f.write((const char*)&decay_rate,  sizeof(float));
        int n = (int)slots_.size();
        f.write((const char*)&n, sizeof(int));
        for (const auto& s : slots_) {
            f.write((const char*)s.vec.data(),
                    (std::streamsize)(s.vec.size() * sizeof(float)));
            f.write((const char*)&s.salience,   sizeof(float));
            f.write((const char*)&s.activation, sizeof(float));
            f.write((const char*)&s.age,        sizeof(int));
        }
    }

    static WorkingMemory load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("WorkingMemory::load: cannot open " + path);
        WorkingMemory wm;
        f.read((char*)&wm.n_dims,     sizeof(int));
        f.read((char*)&wm.capacity,   sizeof(int));
        f.read((char*)&wm.decay_rate, sizeof(float));
        wm.mtx_ = std::make_unique<std::mutex>();
        int n; f.read((char*)&n, sizeof(int));
        wm.slots_.resize(n);
        for (auto& s : wm.slots_) {
            s.vec.resize(wm.n_dims);
            f.read((char*)s.vec.data(),
                   (std::streamsize)(wm.n_dims * sizeof(float)));
            f.read((char*)&s.salience,   sizeof(float));
            f.read((char*)&s.activation, sizeof(float));
            f.read((char*)&s.age,        sizeof(int));
        }
        return wm;
    }
};

} // namespace brain2
