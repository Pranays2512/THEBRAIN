#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace brain2 {

// Operations the BG controller can select
enum class Op : int {
    READ       = 0,  // load from scratchpad slot into working buffer
    WRITE      = 1,  // write working buffer to scratchpad slot
    APPLY      = 2,  // apply symbolic op(slot_a, slot_b) → slot_out
    COMPARE    = 3,  // cosine similarity between two slots
    BIND_QUERY = 4,  // query BindingMemory with (slot_a, slot_b) → slot_out
    RETRIEVE   = 5,  // retrieve from episodic memory
    HALT       = 6,  // done — answer is in "result" slot
    N_OPS      = 7
};

struct BGAction {
    Op          op       = Op::HALT;
    std::string slot_a   = "subject";
    std::string slot_b   = "relation";
    std::string slot_out = "result";
};

// Small 2-layer MLP trained by REINFORCE
struct BasalGanglia {
    int n_dims = 0;
    int hidden = 64;
    int n_ops  = (int)Op::N_OPS;

    // Weights: W1[hidden × 2n_dims], b1[hidden], W2[n_ops × hidden], b2[n_ops]
    std::vector<float> W1, b1, W2, b2;

    float lr_   = 0.001f;
    int   step_ = 0;

    // REINFORCE: track selected op + hidden layer + input for proper gradient
    struct Trace { int op_idx; std::vector<float> h1; std::vector<float> inp; };
    std::vector<Trace> traces_;
    std::mt19937       rng_;

    BasalGanglia() : rng_(42) {}
    BasalGanglia(int n_dims, float lr = 0.001f, unsigned seed = 42)
        : n_dims(n_dims), lr_(lr), rng_(seed) {
        int in = 2 * n_dims;
        std::normal_distribution<float> nd(0.f, 0.02f);
        W1.resize(hidden * in);   for (auto& w : W1) w = nd(rng_);
        b1.resize(hidden, 0.f);
        W2.resize(n_ops * hidden); for (auto& w : W2) w = nd(rng_);
        b2.resize(n_ops, 0.f);
    }

    // Forward: ctx(n_dims) + goal(n_dims) → logits(n_ops)
    std::vector<float> forward(const std::vector<float>& ctx,
                               const std::vector<float>& goal) {
        // Build input — pad/clip to exactly 2*n_dims
        std::vector<float> inp(2 * n_dims, 0.f);
        for (int i = 0; i < n_dims && i < (int)ctx.size();  i++) inp[i]          = ctx[i];
        for (int i = 0; i < n_dims && i < (int)goal.size(); i++) inp[n_dims + i]  = goal[i];

        // Layer 1 — tanh (store h for reinforce)
        std::vector<float> h(hidden, 0.f);
        for (int i = 0; i < hidden; i++) {
            float s = b1[i];
            const float* row = W1.data() + i * 2 * n_dims;
            for (int j = 0; j < 2 * n_dims; j++) s += row[j] * inp[j];
            h[i] = std::tanh(s);
        }

        // Layer 2 — logits
        std::vector<float> logits(n_ops, 0.f);
        for (int i = 0; i < n_ops; i++) {
            float s = b2[i];
            const float* row = W2.data() + i * hidden;
            for (int j = 0; j < hidden; j++) s += row[j] * h[j];
            logits[i] = s;
        }
        return logits; // h not returned here; stored in select_op instead
    }

    static std::vector<float> softmax(const std::vector<float>& x) {
        float mx = *std::max_element(x.begin(), x.end());
        std::vector<float> out(x.size());
        float sum = 0.f;
        for (size_t i = 0; i < x.size(); i++) { out[i] = std::exp(x[i] - mx); sum += out[i]; }
        for (auto& v : out) v /= (sum + 1e-9f);
        return out;
    }

    // Select operation given current scratchpad context + goal
    BGAction select_op(const std::vector<float>& ctx,
                       const std::vector<float>& goal,
                       bool greedy = false) {
        // Rebuild input identical to forward()
        std::vector<float> inp(2 * n_dims, 0.f);
        for (int i = 0; i < n_dims && i < (int)ctx.size();  i++) inp[i]         = ctx[i];
        for (int i = 0; i < n_dims && i < (int)goal.size(); i++) inp[n_dims + i] = goal[i];
        // Layer 1 — recompute h so we can store it for reinforce
        std::vector<float> h(hidden, 0.f);
        for (int i = 0; i < hidden; i++) {
            float s = b1[i];
            const float* row = W1.data() + i * 2 * n_dims;
            for (int j = 0; j < 2 * n_dims; j++) s += row[j] * inp[j];
            h[i] = std::tanh(s);
        }
        auto logits = forward(ctx, goal);
        auto probs  = softmax(logits);
        int chosen;
        if (greedy) {
            chosen = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        } else {
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            chosen = dist(rng_);
        }
        traces_.push_back({chosen, h, inp}); // store h and inp for proper W2/W1 gradient
        BGAction act;
        act.op       = (Op)chosen;
        act.slot_a   = "subject";
        act.slot_b   = "relation";
        act.slot_out = "result";
        return act;
    }

    // REINFORCE: proper policy gradient — update W2, b2, W1, b1
    void reinforce(float reward, bool only_last = true) {
        if (traces_.empty()) return;
        int start = only_last ? traces_.size() - 1 : 0;
        int in = 2 * n_dims;
        for (int i = start; i < (int)traces_.size(); i++) {
            auto& t = traces_[i];
            // Update W2 and b2
            float* row2 = W2.data() + t.op_idx * hidden;
            for (int j = 0; j < hidden; j++)
                row2[j] += lr_ * reward * t.h1[j];
            b2[t.op_idx] += lr_ * reward;

            // Backprop through tanh into W1 and b1
            for (int j = 0; j < hidden; j++) {
                float d = reward * W2[t.op_idx * hidden + j] * (1.f - t.h1[j] * t.h1[j]);
                b1[j] += lr_ * d;
                float* row1 = W1.data() + j * in;
                for (int k = 0; k < in; k++)
                    row1[k] += lr_ * d * t.inp[k];
            }
        }
        traces_.clear();
    }

    void clear_traces() { traces_.clear(); }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("BasalGanglia::save: cannot open " + path);
        f.write((const char*)&n_dims,  sizeof(int));
        f.write((const char*)&hidden,  sizeof(int));
        f.write((const char*)&n_ops,   sizeof(int));
        f.write((const char*)W1.data(), W1.size() * sizeof(float));
        f.write((const char*)b1.data(), b1.size() * sizeof(float));
        f.write((const char*)W2.data(), W2.size() * sizeof(float));
        f.write((const char*)b2.data(), b2.size() * sizeof(float));
    }

    static BasalGanglia load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("BasalGanglia::load: cannot open " + path);
        BasalGanglia bg;
        f.read((char*)&bg.n_dims, sizeof(int));
        f.read((char*)&bg.hidden, sizeof(int));
        f.read((char*)&bg.n_ops,  sizeof(int));
        bg.W1.resize(bg.hidden * 2 * bg.n_dims);
        bg.b1.resize(bg.hidden);
        bg.W2.resize(bg.n_ops * bg.hidden);
        bg.b2.resize(bg.n_ops);
        f.read((char*)bg.W1.data(), bg.W1.size() * sizeof(float));
        f.read((char*)bg.b1.data(), bg.b1.size() * sizeof(float));
        f.read((char*)bg.W2.data(), bg.W2.size() * sizeof(float));
        f.read((char*)bg.b2.data(), bg.b2.size() * sizeof(float));
        return bg;
    }
};

} // namespace brain2
