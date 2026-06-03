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
    READ       = 0,  
    WRITE      = 1,  
    MATH_SUB   = 2,  
    MATH_DIV   = 3,  
    COMPARE    = 4,  
    BIND_QUERY = 5,  
    RETRIEVE   = 6,  
    ANALOGY    = 7,  
    HALT       = 8,  
    STORE_SUBJ = 9,
    STORE_REL  = 10,
    STORE_OBJ  = 11,
    NOT        = 12,
    BIND_ISA   = 13,
    ASK_USER   = 14,
    SPEAK      = 15,
    ATTEND     = 16,
    SPEAK_SUBJ = 17,
    SPEAK_REL  = 18,
    SPEAK_OBJ  = 19,
    MATH_ADD   = 20,
    MATH_MUL   = 21,
    MATH_FACT  = 22,
    MATH_FACT_REL = 23,
    MATH_POW   = 24,
    STORE_TMP  = 25,
    MATH_DIV_FLOAT = 26,
    N_OPS      = 27
};

struct BGAction {
    Op          op       = Op::HALT;
    std::string slot_a   = "subject";
    std::string slot_b   = "relation";
    std::string slot_out = "result";
};

// Small 2-layer MLP trained by TD(lambda) Actor-Critic
struct BasalGanglia {
    int n_dims = 0;
    int hidden = 256;
    int n_ops  = (int)Op::N_OPS;

    // Weights: Actor + Critic
    std::vector<float> W1, b1, W2, b2; // Actor
    std::vector<float> W_v, b_v;       // Critic

    float lr_   = 0.001f;
    int   step_ = 0;

    struct Trace { int op_idx; std::vector<float> h1; std::vector<float> inp; float value; };
    std::vector<Trace> traces_;
    std::mt19937       rng_;

    struct VectorHash {
        size_t operator()(const std::vector<float>& v) const {
            size_t seed = v.size();
            for(auto& i : v) {
                seed ^= std::hash<float>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // Memoization Cache
    std::unordered_map<std::vector<float>, std::pair<std::vector<float>, float>, VectorHash> cache_;
    std::unique_ptr<std::mutex> cache_mtx_;

    BasalGanglia() : rng_(42), cache_mtx_(std::make_unique<std::mutex>()) {}
    BasalGanglia(int n_dims, float lr = 0.001f, unsigned seed = 42)
        : n_dims(n_dims), lr_(lr), rng_(seed), cache_mtx_(std::make_unique<std::mutex>()) {
        int in = 2 * n_dims;
        std::normal_distribution<float> nd(0.f, 0.02f);
        W1.resize(hidden * in);   for (auto& w : W1) w = nd(rng_);
        b1.resize(hidden, 0.f);
        W2.resize(n_ops * hidden); for (auto& w : W2) w = nd(rng_);
        b2.resize(n_ops, 0.f);
        W_v.resize(hidden);        for (auto& w : W_v) w = nd(rng_);
        b_v.resize(1, 0.f);
    }

    // Forward returns (logits, value)
    std::pair<std::vector<float>, float> forward(const std::vector<float>& ctx,
                                                 const std::vector<float>& goal,
                                                 std::vector<float>& h_out,
                                                 std::vector<float>& inp_out) {
        
        {
            std::lock_guard<std::mutex> lock(*cache_mtx_);
            auto it = cache_.find(ctx);
            if (it != cache_.end()) {
                // Must reconstruct inp_out and h_out because trace recording requires them!
                inp_out.assign(2 * n_dims, 0.f);
                for (int i = 0; i < n_dims && i < (int)ctx.size();  i++) inp_out[i]          = ctx[i];
                for (int i = 0; i < n_dims && i < (int)goal.size(); i++) inp_out[n_dims + i]  = goal[i];
                h_out.assign(hidden, 0.f);
                for (int i = 0; i < hidden; i++) {
                    float s = b1[i];
                    const float* row = W1.data() + i * 2 * n_dims;
                    for (int j = 0; j < 2 * n_dims; j++) s += row[j] * inp_out[j];
                    h_out[i] = std::tanh(s);
                }
                return it->second;
            }
        }

        inp_out.assign(2 * n_dims, 0.f);
        for (int i = 0; i < n_dims && i < (int)ctx.size();  i++) inp_out[i]          = ctx[i];
        for (int i = 0; i < n_dims && i < (int)goal.size(); i++) inp_out[n_dims + i]  = goal[i];

        h_out.assign(hidden, 0.f);
        for (int i = 0; i < hidden; i++) {
            float s = b1[i];
            const float* row = W1.data() + i * 2 * n_dims;
            for (int j = 0; j < 2 * n_dims; j++) s += row[j] * inp_out[j];
            h_out[i] = std::tanh(s);
        }

        std::vector<float> logits(n_ops, 0.f);
        for (int i = 0; i < n_ops; i++) {
            float s = b2[i];
            const float* row = W2.data() + i * hidden;
            for (int j = 0; j < hidden; j++) s += row[j] * h_out[j];
            logits[i] = s;
        }

        float value = b_v[0];
        for (int j = 0; j < hidden; j++) value += W_v[j] * h_out[j];

        std::pair<std::vector<float>, float> result = {logits, value};
        {
            std::lock_guard<std::mutex> lock(*cache_mtx_);
            cache_[ctx] = result;
        }
        return result;
    }

    static std::vector<float> softmax(const std::vector<float>& x) {
        float mx = *std::max_element(x.begin(), x.end());
        std::vector<float> out(x.size());
        float sum = 0.f;
        for (size_t i = 0; i < x.size(); i++) { out[i] = std::exp(x[i] - mx); sum += out[i]; }
        for (auto& v : out) v /= (sum + 1e-9f);
        return out;
    }

    void record_trace(int op_idx, const std::vector<float>& h, const std::vector<float>& inp, float value) {
        traces_.push_back({op_idx, h, inp, value});
    }

    BGAction select_op(const std::vector<float>& ctx,
                       const std::vector<float>& goal,
                       bool greedy = false, int force_action = -1) {
        std::vector<float> h, inp;
        auto [logits, value] = forward(ctx, goal, h, inp);
        auto probs  = softmax(logits);
        int chosen;
        if (force_action >= 0) {
            chosen = force_action;
        } else if (greedy) {
            chosen = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        } else {
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            chosen = dist(rng_);
        }
        traces_.push_back({chosen, h, inp, value});
        BGAction act;
        act.op       = (Op)chosen;
        act.slot_a   = "subject";
        act.slot_b   = "relation";
        act.slot_out = "result";
        return act;
    }

    // TD(λ) backward pass
    void reinforce(float final_reward, float gamma = 0.99f, float lambda = 0.95f) {
        if (traces_.empty()) return;
        
        float g_lambda = final_reward;
        int in = 2 * n_dims;

        for (int i = (int)traces_.size() - 1; i >= 0; i--) {
            auto& t = traces_[i];
            float td_error = g_lambda - t.value;

            // Critic Update
            b_v[0] += lr_ * td_error;
            for (int j = 0; j < hidden; j++) {
                W_v[j] += lr_ * td_error * t.h1[j];
            }

            // Actor Update
            float* row2 = W2.data() + t.op_idx * hidden;
            for (int j = 0; j < hidden; j++)
                row2[j] += lr_ * td_error * t.h1[j]; // PG with advantage
            b2[t.op_idx] += lr_ * td_error;

            // Backprop through tanh into W1 and b1 (Actor + Critic)
            for (int j = 0; j < hidden; j++) {
                float d_act = td_error * W2[t.op_idx * hidden + j];
                float d_crit = td_error * W_v[j];
                float d = (d_act + d_crit) * (1.f - t.h1[j] * t.h1[j]);
                
                b1[j] += lr_ * d;
                float* row1 = W1.data() + j * in;
                for (int k = 0; k < in; k++)
                    row1[k] += lr_ * d * t.inp[k];
            }

            // Bootstrap next step
            if (i > 0) {
                g_lambda = 0.f + gamma * ((1.f - lambda) * t.value + lambda * g_lambda);
            }
        }
        traces_.clear();
    }

    void clear_traces() { 
        traces_.clear(); 
        std::lock_guard<std::mutex> lock(*cache_mtx_);
        cache_.clear();
    }

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
        f.write((const char*)W_v.data(), W_v.size() * sizeof(float));
        f.write((const char*)b_v.data(), b_v.size() * sizeof(float));
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
        bg.W_v.resize(bg.hidden);
        bg.b_v.resize(1);
        f.read((char*)bg.W1.data(), bg.W1.size() * sizeof(float));
        f.read((char*)bg.b1.data(), bg.b1.size() * sizeof(float));
        f.read((char*)bg.W2.data(), bg.W2.size() * sizeof(float));
        f.read((char*)bg.b2.data(), bg.b2.size() * sizeof(float));
        f.read((char*)bg.W_v.data(), bg.W_v.size() * sizeof(float));
        f.read((char*)bg.b_v.data(), bg.b_v.size() * sizeof(float));
        
        // Dynamic Expansion if ops were added
        int current_ops = (int)Op::N_OPS;
        if (bg.n_ops < current_ops) {
            std::normal_distribution<float> nd(0.f, 0.02f);
            
            std::vector<float> new_W2(current_ops * bg.hidden, 0.f);
            for (int i = 0; i < bg.n_ops; i++) {
                for (int j = 0; j < bg.hidden; j++) {
                    new_W2[i * bg.hidden + j] = bg.W2[i * bg.hidden + j];
                }
            }
            for (int i = bg.n_ops; i < current_ops; i++) {
                for (int j = 0; j < bg.hidden; j++) {
                    new_W2[i * bg.hidden + j] = nd(bg.rng_);
                }
            }
            
            std::vector<float> new_b2(current_ops, 0.f);
            for (int i = 0; i < bg.n_ops; i++) new_b2[i] = bg.b2[i];
            
            bg.W2 = new_W2;
            bg.b2 = new_b2;
            bg.n_ops = current_ops;
        }
        
        return bg;
    }
};

} // namespace brain2
