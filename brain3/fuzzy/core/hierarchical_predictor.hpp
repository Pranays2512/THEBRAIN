#pragma once
/*
 * hierarchical_predictor.hpp — Multi-timescale LSTM predictor (Brain V3)
 *
 * Three nested levels:
 *   Level 0 (fast)   — 1-step: delegates to existing Predictor in brain.hpp
 *   Level 1 (chunk)  — N-step summaries: learns rules over reasoning chains
 *   Level 2 (episode)— M-chunk summaries: learns episode-level structure
 *
 * Only levels 1 and 2 are defined here; level 0 is the existing Predictor.
 * Error at each level is LOCAL — does not propagate down.
 */
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>
#include <numeric>

namespace brain2 {

// Minimal single-layer LSTM for hierarchical use
struct MiniLSTM {
    int in_dim = 0, h_dim = 0;
    std::vector<float> Wx, Wh, b;   // Wx[4H×I], Wh[4H×H], b[4H]
    std::vector<float> Wo, bo;       // output projection: Wo[I×H], bo[I]
    std::vector<float> h, c;
    
    // Stored state for 1-step BPTT
    std::vector<float> x_prev, h_prev, c_prev;
    std::vector<float> ig, fg, gg, og;

    float last_err = 0.f;

    MiniLSTM() = default;
    MiniLSTM(int in_dim, int h_dim, std::mt19937& rng)
        : in_dim(in_dim), h_dim(h_dim),
          h(h_dim, 0.f), c(h_dim, 0.f),
          x_prev(in_dim, 0.f), h_prev(h_dim, 0.f), c_prev(h_dim, 0.f),
          ig(h_dim, 0.f), fg(h_dim, 0.f), gg(h_dim, 0.f), og(h_dim, 0.f) {
        std::normal_distribution<float> nd(0.f, 0.08f);
        Wx.resize(4*h_dim*in_dim); for (auto& w : Wx) w = nd(rng);
        Wh.resize(4*h_dim*h_dim);  for (auto& w : Wh) w = nd(rng);
        b.resize(4*h_dim, 0.f);
        Wo.resize(in_dim*h_dim);   for (auto& w : Wo) w = nd(rng);
        bo.resize(in_dim, 0.f);
    }

    static float sigmoid_(float x) { return 1.f / (1.f + std::exp(-std::max(-15.f, std::min(15.f, x)))); }

    // Forward one step; compute MSE error against actual if provided
    // Returns prediction of next input (projected back to in_dim)
    std::vector<float> step(const std::vector<float>& x,
                            const std::vector<float>* actual = nullptr,
                            float lr = 0.f) {
        int I = in_dim, H = h_dim;
        x_prev = x;
        h_prev = h;
        c_prev = c;

        // LSTM gates
        std::vector<float> g(4*H, 0.f);
        for (int i = 0; i < 4*H; i++) {
            float s = b[i];
            for (int j = 0; j < I; j++) s += Wx[i*I+j] * x[j];
            for (int j = 0; j < H; j++) s += Wh[i*H+j] * h[j];
            g[i] = s;
        }
        std::vector<float> new_h(H), new_c(H);
        for (int i = 0; i < H; i++) {
            ig[i] = sigmoid_(g[i]);
            fg[i] = sigmoid_(g[H+i]);
            gg[i] = std::tanh(g[2*H+i]);
            og[i] = sigmoid_(g[3*H+i]);
            new_c[i] = fg[i]*c[i] + ig[i]*gg[i];
            new_h[i] = og[i] * std::tanh(new_c[i]);
        }
        h = new_h; c = new_c;

        // Output projection h → prediction [in_dim]
        std::vector<float> pred(I, 0.f);
        for (int i = 0; i < I; i++) {
            float s = bo[i];
            for (int j = 0; j < H; j++) s += Wo[i*H+j] * h[j];
            pred[i] = std::tanh(s);
        }

        // Compute MSE error + 1-step BPTT gradient
        if (actual && lr > 0.f) {
            float err = 0.f;
            std::vector<float> dh(H, 0.f);

            for (int i = 0; i < I; i++) {
                float delta = (*actual)[i] - pred[i];
                err += delta * delta;
                float dpred = delta * (1.f - pred[i]*pred[i]); // tanh deriv
                bo[i] += lr * dpred;
                for (int j = 0; j < H; j++) {
                    float wo = Wo[i*H+j];
                    Wo[i*H+j] += lr * dpred * h[j];
                    dh[j] += dpred * wo;
                }
            }
            last_err = std::sqrt(err / (float)I);

            // Backprop into LSTM core (1 step RTRL approximation)
            for (int i = 0; i < H; i++) {
                float tanh_c = std::tanh(c[i]);
                float d_og = dh[i] * tanh_c * og[i] * (1.f - og[i]);
                float d_c  = dh[i] * og[i] * (1.f - tanh_c * tanh_c);
                float d_ig = d_c * gg[i] * ig[i] * (1.f - ig[i]);
                float d_fg = d_c * c_prev[i] * fg[i] * (1.f - fg[i]);
                float d_gg = d_c * ig[i] * (1.f - gg[i] * gg[i]);

                float d_gates[4] = {d_ig, d_fg, d_gg, d_og};
                for (int gate = 0; gate < 4; gate++) {
                    int row = gate * H + i;
                    b[row] += lr * d_gates[gate];
                    for (int j = 0; j < I; j++) Wx[row*I+j] += lr * d_gates[gate] * x_prev[j];
                    for (int j = 0; j < H; j++) Wh[row*H+j] += lr * d_gates[gate] * h_prev[j];
                }
            }
        }
        return pred;
    }

    void reset() { h.assign(h_dim, 0.f); c.assign(h_dim, 0.f); }

    void save(std::ofstream& f) const {
        f.write((const char*)&in_dim, sizeof(int));
        f.write((const char*)&h_dim,  sizeof(int));
        f.write((const char*)Wx.data(), Wx.size()*sizeof(float));
        f.write((const char*)Wh.data(), Wh.size()*sizeof(float));
        f.write((const char*)b.data(),  b.size()*sizeof(float));
        f.write((const char*)Wo.data(), Wo.size()*sizeof(float));
        f.write((const char*)bo.data(), bo.size()*sizeof(float));
    }

    void load(std::ifstream& f) {
        f.read((char*)&in_dim, sizeof(int));
        f.read((char*)&h_dim,  sizeof(int));
        Wx.resize(4*h_dim*in_dim); Wh.resize(4*h_dim*h_dim);
        b.resize(4*h_dim); Wo.resize(in_dim*h_dim); bo.resize(in_dim);
        h.assign(h_dim, 0.f); c.assign(h_dim, 0.f);
        ig.assign(h_dim, 0.f); fg.assign(h_dim, 0.f);
        gg.assign(h_dim, 0.f); og.assign(h_dim, 0.f);
        f.read((char*)Wx.data(), Wx.size()*sizeof(float));
        f.read((char*)Wh.data(), Wh.size()*sizeof(float));
        f.read((char*)b.data(),  b.size()*sizeof(float));
        f.read((char*)Wo.data(), Wo.size()*sizeof(float));
        f.read((char*)bo.data(), bo.size()*sizeof(float));
    }

    void expand_in_dim(int new_in) {
        if (new_in <= in_dim) return;
        int H = h_dim;
        std::vector<float> new_Wx(4 * H * new_in, 0.f);
        for (int i = 0; i < 4 * H; i++) {
            for (int j = 0; j < in_dim; j++) {
                new_Wx[i * new_in + j] = Wx[i * in_dim + j];
            }
        }
        Wx = std::move(new_Wx);
        
        std::vector<float> new_Wo(new_in * H, 0.f);
        for (int i = 0; i < in_dim; i++) {
            for (int j = 0; j < H; j++) {
                new_Wo[i * H + j] = Wo[i * H + j];
            }
        }
        Wo = std::move(new_Wo);
        bo.resize(new_in, 0.f);
        x_prev.resize(new_in, 0.f);
        in_dim = new_in;
    }
};

struct HierarchicalPredictor {
    // Level 1: operates on summaries of chunk_size fast steps
    MiniLSTM chunk;
    // Level 2: operates on summaries of episode_size chunks
    MiniLSTM episode;

    int chunk_size   = 4;
    int episode_size = 8;
    int n_dims       = 0;
    float lr_chunk   = 0.003f;
    float lr_episode = 0.001f;

    int fast_step_count_  = 0;  // steps since last chunk tick
    int chunk_step_count_ = 0;  // chunks since last episode tick

    // Accumulator for chunk summary (mean of fast-level activations)
    std::vector<float> chunk_acc_;
    std::vector<float> episode_acc_;

    // Delayed targets for autonomous hierarchical learning
    std::vector<float> prev_chunk_summary_;
    std::vector<float> prev_ep_summary_;
    std::vector<float> current_chunk_pred_;
    std::vector<float> current_ep_pred_;
    bool has_prev_chunk_ = false;
    bool has_prev_ep_    = false;

    HierarchicalPredictor() = default;
    HierarchicalPredictor(int n_dims, int chunk_h = 128, int episode_h = 64,
                          unsigned seed = 42)
        : n_dims(n_dims), chunk_acc_(n_dims, 0.f), episode_acc_(n_dims, 0.f),
          prev_chunk_summary_(n_dims, 0.f), prev_ep_summary_(n_dims, 0.f),
          current_chunk_pred_(n_dims, 0.f), current_ep_pred_(n_dims, 0.f) {
        std::mt19937 rng(seed);
        chunk   = MiniLSTM(n_dims, chunk_h,   rng);
        episode = MiniLSTM(n_dims, episode_h, rng);
    }

    // Call every fast step with current SOM activation
    // Internally ticks chunk and episode predictors at their own rates
    void observe(const std::vector<float>& act) {
        // Accumulate into chunk buffer (running mean)
        for (int i = 0; i < n_dims; i++)
            chunk_acc_[i] += act[i];
        fast_step_count_++;

        if (fast_step_count_ >= chunk_size) {
            // Compute chunk summary
            std::vector<float> summary(n_dims);
            for (int i = 0; i < n_dims; i++)
                summary[i] = chunk_acc_[i] / (float)chunk_size;
            chunk_acc_.assign(n_dims, 0.f);
            fast_step_count_ = 0;

            // Step chunk predictor: previous summary should predict THIS summary
            if (has_prev_chunk_) {
                current_chunk_pred_ = chunk.step(prev_chunk_summary_, &summary, lr_chunk);
            }
            prev_chunk_summary_ = summary;
            has_prev_chunk_ = true;

            // Accumulate into episode buffer
            for (int i = 0; i < n_dims; i++)
                episode_acc_[i] += summary[i];
            chunk_step_count_++;

            if (chunk_step_count_ >= episode_size) {
                std::vector<float> ep_summary(n_dims);
                for (int i = 0; i < n_dims; i++)
                    ep_summary[i] = episode_acc_[i] / (float)episode_size;
                episode_acc_.assign(n_dims, 0.f);
                chunk_step_count_ = 0;

                // Step episode predictor: previous episode should predict THIS episode
                if (has_prev_ep_) {
                    current_ep_pred_ = episode.step(prev_ep_summary_, &ep_summary, lr_episode);
                }
                prev_ep_summary_ = ep_summary;
                has_prev_ep_ = true;
            }
        }
    }

    float last_error_chunk()   const { return chunk.last_err;   }
    float last_error_episode() const { return episode.last_err; }

    void reset() {
        chunk.reset();
        episode.reset();
        chunk_acc_.assign(n_dims, 0.f);
        episode_acc_.assign(n_dims, 0.f);
        current_chunk_pred_.assign(n_dims, 0.f);
        current_ep_pred_.assign(n_dims, 0.f);
        fast_step_count_ = 0;
        chunk_step_count_ = 0;
        has_prev_chunk_ = false;
        has_prev_ep_ = false;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("HierarchicalPredictor::save: cannot open " + path);
        f.write((const char*)&n_dims,       sizeof(int));
        f.write((const char*)&chunk_size,   sizeof(int));
        f.write((const char*)&episode_size, sizeof(int));
        chunk.save(f);
        episode.save(f);
    }

    static HierarchicalPredictor load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("HierarchicalPredictor::load: cannot open " + path);
        HierarchicalPredictor hp;
        f.read((char*)&hp.n_dims,       sizeof(int));
        f.read((char*)&hp.chunk_size,   sizeof(int));
        f.read((char*)&hp.episode_size, sizeof(int));
        hp.chunk_acc_.assign(hp.n_dims, 0.f);
        hp.episode_acc_.assign(hp.n_dims, 0.f);
        hp.prev_chunk_summary_.assign(hp.n_dims, 0.f);
        hp.prev_ep_summary_.assign(hp.n_dims, 0.f);
        hp.current_chunk_pred_.assign(hp.n_dims, 0.f);
        hp.current_ep_pred_.assign(hp.n_dims, 0.f);
        hp.chunk.load(f);
        hp.episode.load(f);
        return hp;
    }

    void expand_dims(int new_dims) {
        if (new_dims <= n_dims) return;
        chunk.expand_in_dim(new_dims);
        episode.expand_in_dim(new_dims);
        
        chunk_acc_.resize(new_dims, 0.f);
        episode_acc_.resize(new_dims, 0.f);
        prev_chunk_summary_.resize(new_dims, 0.f);
        prev_ep_summary_.resize(new_dims, 0.f);
        current_chunk_pred_.resize(new_dims, 0.f);
        current_ep_pred_.resize(new_dims, 0.f);
        
        n_dims = new_dims;
    }
};

} // namespace brain2
