#pragma once
/*
 * working_mem.hpp — Hierarchical Spiking Working Memory (Prefrontal Cortex)
 *
 * Implements a rostro-caudal hierarchy of LIF neuron buffers.
 * - Tier 0 (Sensory): Fast decay, high capacity.
 * - Tier 1 (Chunk/Semantic): Medium decay, medium capacity.
 * - Tier 2 (Goal/Executive): Slow decay, low capacity.
 *
 * When a tier fills up, the most stable/salient item being evicted
 * is promoted to the next tier up, preserving abstract context over long timescales.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <string>
#include <limits>

namespace brain2 {

struct LIFSlot {
    std::vector<float> voltage;
    std::vector<float> spike_trace; // low-pass filter of spikes (firing rate)
    float salience;   
    float activity;   
    int   age;

    LIFSlot(int n) {
        voltage.resize(n, 0.f);
        spike_trace.resize(n, 0.f);
        salience = 0.f;
        activity = 0.f;
        age = 0;
    }
    LIFSlot() : LIFSlot(0) {}
};

struct Tier {
    int capacity;
    float decay_rate;
    float trace_decay;
    std::vector<LIFSlot> slots;
    
    Tier(int c, float d, float t) : capacity(c), decay_rate(d), trace_decay(t) {}
    Tier() = default;
};

class WorkingMemory {
public:
    int   n_dims;
    float threshold;  // LIF firing threshold

    std::unique_ptr<std::mutex> mtx_;

    // ── Causal Disruption (Anti-Hebbian Learning) ─────────────
    // When Predictive Coding registers a massive error (surprise),
    // we violently decay the working memory states that caused the bad prediction.
    void disrupt_by_error(const std::vector<float>& error_vec) {
        std::lock_guard<std::mutex> lock(*mtx_);
        float err_norm = norm(error_vec);
        if (err_norm < 0.1f) return; // Only disrupt on significant surprise
        
        for (auto& tier : tiers_) {
            for (auto& s : tier.slots) {
                // How much did this slot contribute to the error direction?
                float correlation = cosine(s.voltage, error_vec);
                if (correlation > 0.2f) { // If it's highly correlated with the error
                    // Violent Hebbian un-learning
                    for (int i = 0; i < n_dims; i++) {
                        s.voltage[i] -= 0.5f * correlation * error_vec[i];
                        s.spike_trace[i] *= 0.5f; // suppress its activity
                    }
                    s.activity *= 0.5f;
                    s.salience *= 0.5f;
                }
            }
        }
    }

private:
    std::vector<Tier> tiers_;

    static float dot(const std::vector<float>& a, const std::vector<float>& b) noexcept {
        float s = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    static float norm(const std::vector<float>& v) noexcept {
        float s = 0.f;
        for (auto x : v) s += x * x;
        return std::sqrt(s);
    }

    static float cosine(const std::vector<float>& a, const std::vector<float>& b) noexcept {
        float na = norm(a), nb = norm(b);
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot(a, b) / (na * nb);
    }

    int least_important(int t_idx) const noexcept {
        int idx = 0;
        float min_val = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)tiers_[t_idx].slots.size(); i++) {
            float val = tiers_[t_idx].slots[i].activity * (1.f + tiers_[t_idx].slots[i].salience);
            if (val < min_val) { min_val = val; idx = i; }
        }
        return idx;
    }
    // Internal gate to a specific tier
    bool gate_tier(int t_idx, const std::vector<float>& activation, float salience, bool is_promotion = false) {
        if (t_idx >= (int)tiers_.size()) return false;
        auto& tier = tiers_[t_idx];
        
        // Merge if similar
        for (auto& s : tier.slots) {
            if (cosine(s.spike_trace, activation) > 0.85f || cosine(s.voltage, activation) > 0.85f) {
                for (int i = 0; i < n_dims; i++) {
                    s.voltage[i] += activation[i] * (is_promotion ? 1.0f : 0.5f);
                }
                s.salience   = std::max(s.salience, salience);
                s.age        = 0;
                return true;
            }
        }

        LIFSlot slot(n_dims);
        for (int i = 0; i < n_dims; i++) {
            slot.voltage[i] = activation[i];
            if (is_promotion) slot.spike_trace[i] = activation[i]; // maintain trace across tiers
        }
        slot.salience   = salience;
        slot.activity   = 1.0f;
        slot.age        = 0;

        if ((int)tier.slots.size() < tier.capacity) {
            tier.slots.push_back(std::move(slot));
        } else {
            int evict_idx = least_important(t_idx);
            auto evicted = std::move(tier.slots[evict_idx]);
            tier.slots[evict_idx] = std::move(slot);
            
            // Promote evicted slot to next tier if it was strong enough
            if (evicted.activity > 0.2f && t_idx + 1 < (int)tiers_.size()) {
                gate_tier(t_idx + 1, evicted.spike_trace, evicted.salience, true);
            }
        }
        return true;
    }

public:
    WorkingMemory() : n_dims(0), threshold(1.0f), mtx_(std::make_unique<std::mutex>()) {}

    // Default hierarchy: 7-slot fast, 4-slot mid, 2-slot slow
    WorkingMemory(int n_dims, int capacity = 7, float decay_rate = 0.95f)
        : n_dims(n_dims), threshold(1.0f), mtx_(std::make_unique<std::mutex>())
    {
        // For backwards compatibility with tests that define capacity and decay_rate, 
        // we use them for Tier 0.
        tiers_.emplace_back(capacity, decay_rate, 0.90f); 
        tiers_.emplace_back(std::max(1, capacity / 2), decay_rate * 1.02f, 0.95f);
        tiers_.emplace_back(std::max(1, capacity / 4), decay_rate * 1.04f, 0.99f);
        
        // Cap decays at 0.995 to avoid infinite potentials
        for (auto& t : tiers_) {
            t.decay_rate = std::min(0.995f, t.decay_rate);
        }
    }

    WorkingMemory(WorkingMemory&&)            = default;
    WorkingMemory& operator=(WorkingMemory&&) = default;
    WorkingMemory(const WorkingMemory&)       = delete;
    WorkingMemory& operator=(const WorkingMemory&) = delete;

    // Gate into Sensory Level 0
    bool gate(const std::vector<float>& activation, float salience = 0.f) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if ((int)activation.size() != n_dims) return false;
        return gate_tier(0, activation, salience, false);
    }

    void tick() {
        std::lock_guard<std::mutex> lock(*mtx_);
        for (int t = 0; t < (int)tiers_.size(); t++) {
            auto& tier = tiers_[t];
            
            // Lateral Inhibition: distinct concept clusters suppress each other
            int n_slots = (int)tier.slots.size();
            std::vector<float> inhibitions(n_slots, 0.f);
            for (int i = 0; i < n_slots; i++) {
                for (int j = 0; j < n_slots; j++) {
                    if (i == j) continue;
                    // Lower similarity means they are competing distinct concepts
                    float sim = cosine(tier.slots[i].spike_trace, tier.slots[j].spike_trace);
                    if (sim < 0.3f) {
                        // J inhibits I based on J's activity
                        inhibitions[i] += tier.slots[j].activity * 0.15f;
                    }
                }
            }

            for (int i = 0; i < n_slots; i++) {
                auto& s = tier.slots[i];
                float total_spikes = 0.f;
                for (int d = 0; d < n_dims; d++) {
                    s.voltage[d] -= inhibitions[i]; // apply inhibition
                    if (s.voltage[d] < 0.f) s.voltage[d] = 0.f;

                    s.voltage[d] *= tier.decay_rate;
                    float spike = 0.f;
                    if (s.voltage[d] > threshold) {
                        spike = 1.0f;
                        s.voltage[d] = 0.0f;
                    }
                    s.spike_trace[d] = s.spike_trace[d] * tier.trace_decay + spike * (1.0f - tier.trace_decay);
                    total_spikes += s.spike_trace[d];
                }
                s.activity = total_spikes / (n_dims + 1e-5f);
                s.age++;
            }
            // Remove dead slots
            tier.slots.erase(
                std::remove_if(tier.slots.begin(), tier.slots.end(),
                    [](const LIFSlot& s){ return s.activity < 0.001f; }),
                tier.slots.end());
        }
    }

    // Context is a hierarchical blend: lower tiers are more heavily weighted for immediate context
    std::vector<float> context() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<float> ctx(n_dims, 0.f);
        float total_weight = 0.f;
        
        for (int t = 0; t < (int)tiers_.size(); t++) {
            float tier_weight = 1.0f / (t + 1.0f); // T0=1.0, T1=0.5, T2=0.33
            const auto& tier = tiers_[t];
            for (const auto& s : tier.slots) {
                float w = (s.activity + 0.1f) * tier_weight;
                for (int i = 0; i < n_dims; i++) ctx[i] += w * s.spike_trace[i];
                total_weight += w;
            }
        }
        
        if (total_weight > 0.f) {
            for (auto& x : ctx) x /= total_weight;
        }
        return ctx;
    }

    // Slot-preserving view: the active slots as SEPARATE vectors (most-active first).
    // context() blends everything into one mean vector — fine for "general gist", fatal
    // for relational reasoning, where holding concept A and concept B DISTINCTLY (not
    // mean(A,B)) is the whole point. This exposes the combinatorial structure.
    std::vector<std::vector<float>> slot_contents(float min_activity = 0.05f) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<std::pair<float, const LIFSlot*>> act;
        for (const auto& t : tiers_)
            for (const auto& s : t.slots)
                if (s.activity > min_activity) act.emplace_back(s.activity, &s);
        std::sort(act.begin(), act.end(),
                  [](const std::pair<float, const LIFSlot*>& a,
                     const std::pair<float, const LIFSlot*>& b) { return a.first > b.first; });
        std::vector<std::vector<float>> out;
        for (const auto& p : act) out.push_back(p.second->voltage);   // the held content per slot
        return out;
    }

    // Most active slot across all tiers
    std::vector<float> most_active() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        const LIFSlot* best = nullptr;
        for (const auto& t : tiers_) {
            for (const auto& s : t.slots) {
                if (!best || s.activity > best->activity) best = &s;
            }
        }
        return best ? best->spike_trace : std::vector<float>(n_dims, 0.f);
    }

    void boost_salience(const std::vector<float>& v, float amount) {
        std::lock_guard<std::mutex> lock(*mtx_);
        float best_sim = -1.f;
        LIFSlot* best_slot = nullptr;
        
        for (auto& t : tiers_) {
            for (auto& s : t.slots) {
                float sim = cosine(s.spike_trace, v);
                if (sim > best_sim) { best_sim = sim; best_slot = &s; }
            }
        }
        if (best_slot) {
            best_slot->salience = std::min(1.f, best_slot->salience + amount);
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(*mtx_);
        for (auto& t : tiers_) t.slots.clear();
    }

    int size() const noexcept { 
        int sz = 0;
        for (const auto& t : tiers_) sz += (int)t.slots.size();
        return sz; 
    }
    
    // Specifically returning tier 0 capacity to maintain python test assumptions
    int get_base_capacity() const noexcept {
        return tiers_.empty() ? 0 : tiers_[0].capacity;
    }

    bool empty() const noexcept { return size() == 0; }

    std::vector<float> activations() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<float> a;
        for (const auto& t : tiers_) {
            for (const auto& s : t.slots) a.push_back(s.activity);
        }
        return a;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("WorkingMemory::save: cannot open " + path);
        f.write((const char*)&n_dims, sizeof(int));
        int nt = (int)tiers_.size();
        f.write((const char*)&nt, sizeof(int));
        
        for (const auto& t : tiers_) {
            f.write((const char*)&t.capacity, sizeof(int));
            f.write((const char*)&t.decay_rate, sizeof(float));
            f.write((const char*)&t.trace_decay, sizeof(float));
            int ns = (int)t.slots.size();
            f.write((const char*)&ns, sizeof(int));
            for (const auto& s : t.slots) {
                f.write((const char*)s.voltage.data(),     (std::streamsize)(n_dims * sizeof(float)));
                f.write((const char*)s.spike_trace.data(), (std::streamsize)(n_dims * sizeof(float)));
                f.write((const char*)&s.salience, sizeof(float));
                f.write((const char*)&s.activity, sizeof(float));
                f.write((const char*)&s.age,      sizeof(int));
            }
        }
    }

    static WorkingMemory load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("WorkingMemory::load: cannot open " + path);
        WorkingMemory wm;
        f.read((char*)&wm.n_dims, sizeof(int));
        wm.threshold = 1.0f;
        wm.mtx_ = std::make_unique<std::mutex>();
        
        int nt; f.read((char*)&nt, sizeof(int));
        wm.tiers_.resize(nt);
        for (auto& t : wm.tiers_) {
            f.read((char*)&t.capacity, sizeof(int));
            f.read((char*)&t.decay_rate, sizeof(float));
            f.read((char*)&t.trace_decay, sizeof(float));
            int ns; f.read((char*)&ns, sizeof(int));
            t.slots.resize(ns, LIFSlot(wm.n_dims));
            for (auto& s : t.slots) {
                f.read((char*)s.voltage.data(),     (std::streamsize)(wm.n_dims * sizeof(float)));
                f.read((char*)s.spike_trace.data(), (std::streamsize)(wm.n_dims * sizeof(float)));
                f.read((char*)&s.salience, sizeof(float));
                f.read((char*)&s.activity, sizeof(float));
                f.read((char*)&s.age,      sizeof(int));
            }
        }
        return wm;
    }

    void expand_dims(int new_dims) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (new_dims <= n_dims) return;
        for (auto& t : tiers_) {
            for (auto& s : t.slots) {
                s.voltage.resize(new_dims, 0.f);
                s.spike_trace.resize(new_dims, 0.f);
            }
        }
        n_dims = new_dims;
    }
};

} // namespace brain2
