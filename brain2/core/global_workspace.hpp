#pragma once
#include <vector>
#include <algorithm>

namespace brain2 {

// Integer IDs for modules that can bid for the global broadcast
enum class GWModule : int {
    SOM      = 0,
    PREDICT  = 1,
    LANGUAGE = 2,
    EPISODIC = 3,
    BINDING  = 4,
    EMOTION  = 5,
};

struct GlobalWorkspace {
    struct Bid {
        int                module_id;
        float              salience;
        std::vector<float> representation;
    };

    int n_dims;
    int winner_id_ = -1;
    std::vector<Bid>   bids_;
    std::vector<float> broadcast_;

    GlobalWorkspace() : n_dims(0) {}
    explicit GlobalWorkspace(int n_dims) : n_dims(n_dims), broadcast_(n_dims, 0.f) {}

    // Each module submits its salience + current representation
    void bid(int module_id, float salience, const std::vector<float>& rep) {
        bids_.push_back({module_id, salience, rep});
    }

    // Runs competition — highest salience wins, losers are suppressed
    // Returns winning module_id (-1 if no bids)
    int compete() {
        if (bids_.empty()) { winner_id_ = -1; return -1; }
        auto it = std::max_element(bids_.begin(), bids_.end(),
            [](const Bid& a, const Bid& b){ return a.salience < b.salience; });
        winner_id_ = it->module_id;
        broadcast_ = it->representation;
        bids_.clear();
        return winner_id_;
    }

    // Check if a module won (call after compete())
    bool is_winner(int module_id) const { return module_id == winner_id_; }

    const std::vector<float>& broadcast() const { return broadcast_; }
    int winner_id() const { return winner_id_; }
};

} // namespace brain2
