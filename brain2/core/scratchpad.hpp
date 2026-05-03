#pragma once
/*
 * scratchpad.hpp — Scratchpad Memory, Component 11 of Brain v2
 *
 * External memory tape for deliberate, step-by-step reasoning.
 * Bypasses Working Memory's 7-slot limit entirely.
 *
 * Like paper for a human doing math:
 *   - Write intermediate results to named slots
 *   - Read them back in later steps
 *   - No decay, no capacity limit, no interference with WM
 *
 * Also supports:
 *   - Stack (push/pop) for recursive reasoning
 *   - History per slot (last N writes) for backtracking
 *   - Slot tagging (what kind of value is stored here)
 *   - Diff: compare two slots (are they similar?)
 */

#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <memory>
#include <fstream>
#include <stdexcept>

namespace brain2 {

struct ScratchSlot {
    std::vector<float>              value;      // current value
    std::deque<std::vector<float>>  history;    // last N values
    std::string                     tag;        // "number", "result", "premise", etc.
    int                             write_count;
    static constexpr int            MAX_HISTORY = 8;
};

class Scratchpad {
public:
    int n_dims;

private:
    std::unordered_map<std::string, ScratchSlot> slots_;
    std::vector<std::vector<float>>              stack_;  // push/pop stack
    std::vector<std::string>                     write_order_; // insertion order
    std::unique_ptr<std::mutex>                  mtx_;

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

public:
    Scratchpad() : n_dims(0), mtx_(std::make_unique<std::mutex>()) {}

    Scratchpad(int n_dims) : n_dims(n_dims),
                              mtx_(std::make_unique<std::mutex>()) {}

    Scratchpad(Scratchpad&&)            = default;
    Scratchpad& operator=(Scratchpad&&) = default;
    Scratchpad(const Scratchpad&)       = delete;
    Scratchpad& operator=(const Scratchpad&) = delete;

    // Write concept vector to named slot
    void write(const std::string& name,
               const std::vector<float>& vec,
               const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto& slot = slots_[name];
        if (slot.write_count == 0) write_order_.push_back(name);
        // Save history before overwrite
        if (!slot.value.empty()) {
            slot.history.push_back(slot.value);
            if ((int)slot.history.size() > ScratchSlot::MAX_HISTORY)
                slot.history.pop_front();
        }
        slot.value = vec;
        if (!tag.empty()) slot.tag = tag;
        slot.write_count++;
    }

    // Read concept vector from named slot
    // Returns zero vector if slot doesn't exist
    std::vector<float> read(const std::string& name) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = slots_.find(name);
        if (it == slots_.end()) return std::vector<float>(n_dims, 0.f);
        return it->second.value;
    }

    // Check if slot exists and has a value
    bool has(const std::string& name) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return slots_.count(name) > 0 && !slots_.at(name).value.empty();
    }

    // Read previous value (one step back in history)
    std::vector<float> read_prev(const std::string& name) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = slots_.find(name);
        if (it == slots_.end() || it->second.history.empty())
            return std::vector<float>(n_dims, 0.f);
        return it->second.history.back();
    }

    // How similar are two slots? (for convergence check)
    float similarity(const std::string& a, const std::string& b) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto ia = slots_.find(a);
        auto ib = slots_.find(b);
        if (ia == slots_.end() || ib == slots_.end()) return 0.f;
        return cosine(ia->second.value, ib->second.value);
    }

    // How much did slot change on last write? (convergence signal)
    float delta(const std::string& name) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = slots_.find(name);
        if (it == slots_.end() || it->second.history.empty()) return 1.f;
        return 1.f - cosine(it->second.value, it->second.history.back());
    }

    // Stack operations (for recursive reasoning)
    void push(const std::vector<float>& vec) {
        std::lock_guard<std::mutex> lock(*mtx_);
        stack_.push_back(vec);
    }

    std::vector<float> pop() {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (stack_.empty()) return std::vector<float>(n_dims, 0.f);
        auto v = stack_.back();
        stack_.pop_back();
        return v;
    }

    std::vector<float> peek() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (stack_.empty()) return std::vector<float>(n_dims, 0.f);
        return stack_.back();
    }

    // Copy one slot to another
    void copy(const std::string& src, const std::string& dst) {
        auto v = read(src);
        write(dst, v);
    }

    // Accumulate: slot[name] = blend(slot[name], vec, alpha)
    void accumulate(const std::string& name,
                    const std::vector<float>& vec,
                    float alpha = 0.5f) {
        auto current = read(name);
        std::vector<float> blended(n_dims, 0.f);
        for (int i = 0; i < n_dims && i < (int)vec.size(); i++)
            blended[i] = (1.f - alpha) * current[i] + alpha * vec[i];
        write(name, blended);
    }

    // List all slot names in write order
    std::vector<std::string> slot_names() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return write_order_;
    }

    // Delete a slot
    void erase(const std::string& name) {
        std::lock_guard<std::mutex> lock(*mtx_);
        slots_.erase(name);
        write_order_.erase(
            std::remove(write_order_.begin(), write_order_.end(), name),
            write_order_.end());
    }

    // Clear everything (new problem)
    void clear() {
        std::lock_guard<std::mutex> lock(*mtx_);
        slots_.clear();
        stack_.clear();
        write_order_.clear();
    }

    int slot_count() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return (int)slots_.size();
    }

    int stack_size() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return (int)stack_.size();
    }

    std::string tag(const std::string& name) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = slots_.find(name);
        return it == slots_.end() ? "" : it->second.tag;
    }

    int write_count(const std::string& name) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = slots_.find(name);
        return it == slots_.end() ? 0 : it->second.write_count;
    }
};

} // namespace brain2
