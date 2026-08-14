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
    PREDICT_WM = 27,
    CHAIN_FOLLOW = 28,   // iterative BFS along a relation (multi-hop causal)
    MATH_POLY  = 29,     // evaluate polynomial
    MATH_QUAD  = 30,     // find roots of quadratic equation
    N_OPS      = 31
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

    // ── Experience Replay Buffer (prevents catastrophic forgetting) ───────────
    struct Experience {
        std::vector<float> ctx;   // scratchpad context at decision time
        std::vector<float> goal;  // goal vector
        int                op;    // chosen op
        float              reward;
    };
    static constexpr int REPLAY_CAPACITY = 500;
    static constexpr int REPLAY_BATCH    = 2;
    std::vector<Experience> replay_buffer_;
    int                     replay_head_ = 0;  // ring-buffer write pointer

    void push_experience(const std::vector<float>& ctx,
                         const std::vector<float>& goal,
                         int op, float reward) {
        if ((int)replay_buffer_.size() < REPLAY_CAPACITY) {
            replay_buffer_.push_back({ctx, goal, op, reward});
        } else {
            replay_buffer_[replay_head_] = {ctx, goal, op, reward};
            replay_head_ = (replay_head_ + 1) % REPLAY_CAPACITY;
        }
    }

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
        int in = 5 * n_dims;
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
                inp_out.assign(5 * n_dims, 0.f);
                for (int i = 0; i < 4 * n_dims && i < (int)ctx.size();  i++) inp_out[i]              = ctx[i];
                for (int i = 0; i < n_dims && i < (int)goal.size();     i++) inp_out[4 * n_dims + i] = goal[i];
                h_out.assign(hidden, 0.f);
                for (int i = 0; i < hidden; i++) {
                    float s = b1[i];
                    const float* row = W1.data() + i * 5 * n_dims;
                    for (int j = 0; j < 5 * n_dims; j++) s += row[j] * inp_out[j];
                    h_out[i] = std::tanh(s);
                }
                return it->second;
            }
        }

        inp_out.assign(5 * n_dims, 0.f);
        for (int i = 0; i < 4 * n_dims && i < (int)ctx.size();  i++) inp_out[i]              = ctx[i];
        for (int i = 0; i < n_dims && i < (int)goal.size();     i++) inp_out[4 * n_dims + i] = goal[i];

        h_out.assign(hidden, 0.f);
        for (int i = 0; i < hidden; i++) {
            float s = b1[i];
            const float* row = W1.data() + i * 5 * n_dims;
            for (int j = 0; j < 5 * n_dims; j++) s += row[j] * inp_out[j];
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

    // TD(λ) backward pass + experience replay
    // One actor-critic gradient step for a (op, activations, input) under an advantage.
    void apply_grad(int op_idx, const std::vector<float>& h1,
                    const std::vector<float>& inp, float td_error) {
        td_error = std::max(-2.0f, std::min(2.0f, td_error));   // clip for stability
        int in = 5 * n_dims;
        b_v[0] += lr_ * td_error;
        for (int j = 0; j < hidden; j++)
            W_v[j] += lr_ * td_error * h1[j] - 0.005f * lr_ * W_v[j];
        float* row2 = W2.data() + op_idx * hidden;
        for (int j = 0; j < hidden; j++)
            row2[j] += lr_ * td_error * h1[j] - 0.005f * lr_ * row2[j];
        b2[op_idx] += lr_ * td_error;
        for (int j = 0; j < hidden; j++) {
            float d = (td_error * W2[op_idx * hidden + j] + td_error * W_v[j]) * (1.f - h1[j] * h1[j]);
            b1[j] += lr_ * d;
            float* row1 = W1.data() + j * in;
            for (int k = 0; k < in; k++)
                row1[k] += lr_ * d * inp[k] - 0.005f * lr_ * row1[k];
        }
    }

    // PER-STEP credit: each op gets its OWN reward (advantage = r_i - value_i), so the op
    // that earned the reward is credited and its episode-mates are not. The single-scalar
    // reinforce() spread one reward to all ops -> no differential preference -> the policy
    // couldn't actually prefer the rewarded op. This is the fix for stable consolidation.
    void reinforce_steps(const std::vector<float>& step_rewards) {
        if (traces_.empty()) return;
        int n = std::min((int)traces_.size(), (int)step_rewards.size());
        for (int i = 0; i < n; i++) {
            auto& t = traces_[i];
            apply_grad(t.op_idx, t.h1, t.inp, step_rewards[i] - t.value);
            push_experience(t.inp, {}, t.op_idx, step_rewards[i]);
        }
        traces_.clear();
        std::lock_guard<std::mutex> lock(*cache_mtx_);
        cache_.clear();
    }

    void reinforce(float final_reward, float gamma = 0.99f, float lambda = 0.95f) {
        if (traces_.empty()) return;

        // Store EVERY step for replay. The old code stored only traces_[0] (the first op),
        // so replay endlessly reinforced stale FIRST-step actions with old rewards while the
        // policy moved on -> a destabilizing pull. Replay should sample real (state,op) pairs.
        for (const auto& t : traces_) {
            push_experience(t.inp, {}, t.op_idx, final_reward);
        }

        float g_lambda = final_reward;
        // On-policy trace update
        for (int i = (int)traces_.size() - 1; i >= 0; i--) {
            auto& t = traces_[i];
            apply_grad(t.op_idx, t.h1, t.inp, g_lambda - t.value);
            if (i > 0)
                g_lambda = gamma * ((1.f - lambda) * t.value + lambda * g_lambda);
        }
        traces_.clear();

        // ── Replay: sample REPLAY_BATCH old experiences to fight forgetting ──
        if ((int)replay_buffer_.size() >= REPLAY_BATCH * 2) {
            std::uniform_int_distribution<int> pick(0, (int)replay_buffer_.size() - 1);
            for (int s = 0; s < REPLAY_BATCH; s++) {
                auto& exp = replay_buffer_[pick(rng_)];
                // Recompute h for this experience
                int in = 5 * n_dims;
                std::vector<float> h_r(hidden, 0.f);
                for (int i = 0; i < hidden; i++) {
                    float sv = b1[i];
                    const float* row = W1.data() + i * in;
                    for (int k = 0; k < in; k++) sv += row[k] * exp.ctx[k];
                    h_r[i] = std::tanh(sv);
                }
                float val_r = b_v[0];
                for (int j = 0; j < hidden; j++) val_r += W_v[j] * h_r[j];
                float td_r = exp.reward - val_r;
                // Use smaller lr for replay to avoid overwriting current policy
                float saved_lr = lr_;
                lr_ *= 0.3f;
                apply_grad(exp.op, h_r, exp.ctx, td_r);
                lr_ = saved_lr;
            }
        }

        // THE BUG FIX: weights just changed, so every memoized forward() output is now STALE.
        // Without this, the cache kept serving the OLD logits/value for any context seen
        // before -> the policy stayed frozen across episodes (narrow op set, flat reward)
        // even though reinforce was correctly updating the weights. Clear it so the next
        // decision recomputes with the updated policy.
        {
            std::lock_guard<std::mutex> lock(*cache_mtx_);
            cache_.clear();
        }
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
        bg.W1.resize(bg.hidden * 5 * bg.n_dims);
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

    void expand_dims(int new_dims) {
        std::lock_guard<std::mutex> lock(*cache_mtx_);
        if (new_dims <= n_dims) return;
        
        int old_in = 5 * n_dims;
        int new_in = 5 * new_dims;
        std::vector<float> new_W1(hidden * new_in, 0.f);
        
        for (int i = 0; i < hidden; i++) {
            // Context part
            for (int j = 0; j < n_dims; j++) {
                new_W1[i * new_in + j] = W1[i * old_in + j];
            }
            // Goal part
            for (int j = 0; j < n_dims; j++) {
                new_W1[i * new_in + new_dims + j] = W1[i * old_in + n_dims + j];
            }
        }
        W1 = std::move(new_W1);
        
        for (auto& exp : replay_buffer_) {
            exp.ctx.resize(new_dims, 0.f);
            exp.goal.resize(new_dims, 0.f);
        }
        
        for (auto& tr : traces_) {
            std::vector<float> new_inp(new_in, 0.f);
            for (int j = 0; j < n_dims; j++) {
                new_inp[j] = tr.inp[j];
                new_inp[new_dims + j] = tr.inp[n_dims + j];
            }
            tr.inp = std::move(new_inp);
        }
        
        cache_.clear();
        n_dims = new_dims;
    }
};

} // namespace brain2
