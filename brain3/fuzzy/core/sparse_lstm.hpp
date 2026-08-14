#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <deque>
#include <algorithm>
#include <iostream>

#include "cuda_math.cuh"

namespace brain2 {

// ── Sparse LSH Router ──────────────────────────────────────────────────

struct LSHRouter {
    int input_dim;
    int num_hyperplanes;
    std::vector<std::vector<float>> hyperplanes;
    std::vector<std::vector<int>> bucket_to_neurons;

    LSHRouter() = default;

    LSHRouter(int in_dim, int planes, int total_neurons, int neurons_per_bucket, std::mt19937& rng)
        : input_dim(in_dim), num_hyperplanes(planes) {
        
        // Initialize random hyperplanes
        std::normal_distribution<float> dist(0.f, 1.f);
        hyperplanes.resize(num_hyperplanes, std::vector<float>(input_dim));
        for (int i = 0; i < num_hyperplanes; i++) {
            for (int j = 0; j < input_dim; j++) {
                hyperplanes[i][j] = dist(rng);
            }
        }

        // Initialize Buckets
        int num_buckets = 1 << num_hyperplanes;
        bucket_to_neurons.resize(num_buckets);
        
        // Randomly assign neurons to buckets
        // Over time, Hebbian learning will naturally align the weights of these neurons 
        // to the semantic regions defined by their assigned LSH bucket.
        std::uniform_int_distribution<int> n_dist(0, total_neurons - 1);
        for (int b = 0; b < num_buckets; b++) {
            std::vector<int> active;
            for(int k=0; k < neurons_per_bucket; k++) {
                active.push_back(n_dist(rng));
            }
            // Sort and unique to prevent duplicates
            std::sort(active.begin(), active.end());
            active.erase(std::unique(active.begin(), active.end()), active.end());
            bucket_to_neurons[b] = active;
        }
    }

    int get_bucket(const std::vector<float>& x) const {
        int bucket = 0;
        for (int i = 0; i < num_hyperplanes; i++) {
            float dot = 0.f;
            for (int j = 0; j < input_dim; j++) dot += hyperplanes[i][j] * x[j];
            if (dot > 0) bucket |= (1 << i);
        }
        return bucket;
    }
    
    void save(std::ofstream& f) const {
        f.write((const char*)&input_dim, sizeof(int));
        f.write((const char*)&num_hyperplanes, sizeof(int));
        for(int i=0; i<num_hyperplanes; i++) {
            f.write((const char*)hyperplanes[i].data(), input_dim * sizeof(float));
        }
        int num_buckets = 1 << num_hyperplanes;
        for(int b=0; b<num_buckets; b++) {
            int sz = bucket_to_neurons[b].size();
            f.write((const char*)&sz, sizeof(int));
            if(sz > 0) {
                f.write((const char*)bucket_to_neurons[b].data(), sz * sizeof(int));
            }
        }
    }
    
    static LSHRouter load(std::ifstream& f) {
        LSHRouter r;
        f.read((char*)&r.input_dim, sizeof(int));
        f.read((char*)&r.num_hyperplanes, sizeof(int));
        r.hyperplanes.resize(r.num_hyperplanes, std::vector<float>(r.input_dim));
        for(int i=0; i<r.num_hyperplanes; i++) {
            f.read((char*)r.hyperplanes[i].data(), r.input_dim * sizeof(float));
        }
        int num_buckets = 1 << r.num_hyperplanes;
        r.bucket_to_neurons.resize(num_buckets);
        for(int b=0; b<num_buckets; b++) {
            int sz;
            f.read((char*)&sz, sizeof(int));
            if(sz > 0) {
                r.bucket_to_neurons[b].resize(sz);
                f.read((char*)r.bucket_to_neurons[b].data(), sz * sizeof(int));
            }
        }
        return r;
    }
};

// ── Sparse Forward snapshot ──────────────────────────────────────────

struct SparseForwardSnapshot {
    std::vector<float> x;      
    std::vector<float> h_prev; 
    std::vector<float> c_prev; 
    std::vector<float> gates;  
    std::vector<float> c_new;  
    std::vector<int> active_neurons; // The specific neurons awake at this step
};

// ── Sparse LSTM Layer ────────────────────────────────────────────────

struct SparseLSTMLayer {
    int input_dim, hidden_dim;
    int k_active;
    float weight_decay = 0.f; // decoupled L2 on Wx/Wh active rows (0 = off)

    DeviceVector<float> Wh, Wx, b; // Dense backing matrices
    DeviceVector<float> h, c;      // Current state

    LSHRouter router;
    std::deque<SparseForwardSnapshot> history_;

    SparseLSTMLayer() = default;

    SparseLSTMLayer(int input_dim, int hidden_dim, int k_active, std::mt19937& rng)
        : input_dim(input_dim), hidden_dim(hidden_dim), k_active(k_active) {
        
        int H = hidden_dim, I = input_dim;
        float scale_h = std::sqrt(1.f / H);
        float scale_x = std::sqrt(2.f / (I + H));
        std::normal_distribution<float> dh(0.f, scale_h);
        std::normal_distribution<float> dx(0.f, scale_x);

        Wh.resize(4 * H * H);
        Wx.resize(4 * H * I);
        b.resize(4 * H, 0.f);
        h.resize(H, 0.f);
        c.resize(H, 0.f);

        for (auto& w : Wh) w = dh(rng);
        for (auto& w : Wx) w = dx(rng);
        for (int i = H; i < 2 * H; i++) b[i] = 1.0f; // Forget gate bias = 1.0

        // Create LSH Router: 8 hyperplanes = 256 buckets. 
        // Each bucket wakes up a sparse subset of neurons.
        router = LSHRouter(input_dim, 8, hidden_dim, k_active, rng);
    }

    SparseLSTMLayer(SparseLSTMLayer&&) = default;
    SparseLSTMLayer& operator=(SparseLSTMLayer&&) = default;

    std::vector<float> forward(const std::vector<float>& x, bool record_history = true) {
        int H = hidden_dim, I = input_dim;
        
        int bucket = router.get_bucket(x);
        std::vector<int> active = router.bucket_to_neurons[bucket];
        
        // Small World Cross-Talk: Add 5% random neurons to ensure strict buckets can communicate
        // mathematically allowing complex analogies across disjoint semantic regions.
        int num_random = std::max(1, (int)(H * 0.05f));
        std::uniform_int_distribution<int> dist(0, H - 1);
        std::mt19937 temp_rng(bucket + (int)x[0]*1000); // Deterministic randomness for consistency
        for(int i=0; i<num_random; i++) {
            active.push_back(dist(temp_rng));
        }
        std::sort(active.begin(), active.end());
        active.erase(std::unique(active.begin(), active.end()), active.end());

        std::vector<float> gates(4 * H, 0.f);
        std::vector<float> h_prev(h.begin(), h.end());
        std::vector<float> c_prev(c.begin(), c.end());
        
        std::vector<float> h_new = h_prev;
        std::vector<float> c_new = c_prev;

        // ONLY compute gates for the ACTIVE subset of neurons (O(1) instead of O(N))
        for (int n : active) {
            float fg=0.f, ig=0.f, gg=0.f, og=0.f;
            
            // Sparse MatVec for just this neuron's row
            for (int j = 0; j < I; j++) {
                float xj = x[j];
                fg += Wx[(n) * I + j] * xj;
                ig += Wx[(H + n) * I + j] * xj;
                gg += Wx[(2 * H + n) * I + j] * xj;
                og += Wx[(3 * H + n) * I + j] * xj;
            }
            for (int j = 0; j < H; j++) {
                float hj = h_prev[j]; // Full cross-talk! Neuron reads all 512 prev hidden states
                fg += Wh[(n) * H + j] * hj;
                ig += Wh[(H + n) * H + j] * hj;
                gg += Wh[(2 * H + n) * H + j] * hj;
                og += Wh[(3 * H + n) * H + j] * hj;
            }
            
            fg += b[n];
            ig += b[H + n];
            gg += b[2 * H + n];
            og += b[3 * H + n];

            // Activations
            fg = 1.f / (1.f + std::exp(-fg));
            ig = 1.f / (1.f + std::exp(-ig));
            gg = std::tanh(gg);
            og = 1.f / (1.f + std::exp(-og));

            gates[n] = fg;
            gates[H + n] = ig;
            gates[2 * H + n] = gg;
            gates[3 * H + n] = og;

            c_new[n] = fg * c_prev[n] + ig * gg;
            h_new[n] = og * std::tanh(c_new[n]);
        }

        // Apply state updates
        std::copy(h_new.begin(), h_new.end(), h.begin());
        std::copy(c_new.begin(), c_new.end(), c.begin());

        if (record_history) {
            SparseForwardSnapshot snap;
            snap.x = x;
            snap.h_prev = h_prev;
            snap.c_prev = c_prev;
            snap.gates = gates;
            snap.c_new = c_new;
            snap.active_neurons = active;
            history_.push_back(std::move(snap));
        }

        return h_new;
    }

    std::vector<std::vector<float>> backward_through_time(const std::vector<std::vector<float>>& delta_h_seq, float lr, int n_steps = -1) {
        int H = hidden_dim, I = input_dim;
        int hist = (int)history_.size();
        int steps = (n_steps < 0) ? hist : std::min(n_steps, hist);
        if (steps == 0) return {};

        std::vector<float> delta_h(H, 0.f);
        std::vector<std::vector<float>> delta_x_seq(steps, std::vector<float>(I, 0.f));

        for (int s = steps - 1; s >= 0; s--) {
            int hist_idx = hist - steps + s;
            const SparseForwardSnapshot& snap = history_[hist_idx];

            if (s < (int)delta_h_seq.size()) {
                for (int k = 0; k < H; k++) delta_h[k] += delta_h_seq[s][k];
            }

            const float* fg = snap.gates.data();
            const float* ig = snap.gates.data() + H;
            const float* gg = snap.gates.data() + 2 * H;
            const float* og = snap.gates.data() + 3 * H;

            std::vector<float> dpre(4 * H, 0.f);
            
            // Only backpropagate through the neurons that were ACTIVE at this step
            for (int n : snap.active_neurons) {
                float tanh_c = std::tanh(snap.c_new[n]);
                float dO = delta_h[n] * tanh_c;
                float dC = delta_h[n] * og[n] * (1.f - tanh_c * tanh_c);
                float dI = dC * gg[n];
                float dG = dC * ig[n];
                float dF = dC * snap.c_prev[n];

                dpre[n] = dF * fg[n] * (1.f - fg[n]);
                dpre[H + n] = dI * ig[n] * (1.f - ig[n]);
                dpre[2 * H + n] = dG * (1.f - gg[n] * gg[n]);
                dpre[3 * H + n] = dO * og[n] * (1.f - og[n]);
                
                // Clip gradients
                for(int i=0; i<=3; i++) {
                    if (dpre[i*H + n] > 5.f) dpre[i*H + n] = 5.f;
                    if (dpre[i*H + n] < -5.f) dpre[i*H + n] = -5.f;
                }
            }

            std::vector<float> delta_x(I, 0.f);
            for (int n : snap.active_neurons) {
                for (int j = 0; j < I; j++) {
                    delta_x[j] += Wx[n * I + j] * dpre[n] + 
                                  Wx[(H + n) * I + j] * dpre[H + n] + 
                                  Wx[(2 * H + n) * I + j] * dpre[2 * H + n] + 
                                  Wx[(3 * H + n) * I + j] * dpre[3 * H + n];
                }
            }

            std::vector<float> delta_h_prev(H, 0.f);
            for (int n : snap.active_neurons) {
                for (int j = 0; j < H; j++) {
                    delta_h_prev[j] += Wh[n * H + j] * dpre[n] + 
                                       Wh[(H + n) * H + j] * dpre[H + n] + 
                                       Wh[(2 * H + n) * H + j] * dpre[2 * H + n] + 
                                       Wh[(3 * H + n) * H + j] * dpre[3 * H + n];
                }
            }

            // Weight updates ONLY for the active rows!
            // Decoupled weight decay (AdamW-style): shrink active rows toward 0
            // before the gradient step. Bias is never decayed.
            float wd = (weight_decay > 0.f) ? (1.f - lr * weight_decay) : 1.f;
            for (int n : snap.active_neurons) {
                for (int j = 0; j < I; j++) {
                    float xj = snap.x[j];
                    Wx[n * I + j] = wd * Wx[n * I + j] - lr * dpre[n] * xj;
                    Wx[(H + n) * I + j] = wd * Wx[(H + n) * I + j] - lr * dpre[H + n] * xj;
                    Wx[(2 * H + n) * I + j] = wd * Wx[(2 * H + n) * I + j] - lr * dpre[2 * H + n] * xj;
                    Wx[(3 * H + n) * I + j] = wd * Wx[(3 * H + n) * I + j] - lr * dpre[3 * H + n] * xj;
                }
                for (int j = 0; j < H; j++) {
                    float hj = snap.h_prev[j];
                    Wh[n * H + j] = wd * Wh[n * H + j] - lr * dpre[n] * hj;
                    Wh[(H + n) * H + j] = wd * Wh[(H + n) * H + j] - lr * dpre[H + n] * hj;
                    Wh[(2 * H + n) * H + j] = wd * Wh[(2 * H + n) * H + j] - lr * dpre[2 * H + n] * hj;
                    Wh[(3 * H + n) * H + j] = wd * Wh[(3 * H + n) * H + j] - lr * dpre[3 * H + n] * hj;
                }
                b[n] -= lr * dpre[n];
                b[H + n] -= lr * dpre[H + n];
                b[2 * H + n] -= lr * dpre[2 * H + n];
                b[3 * H + n] -= lr * dpre[3 * H + n];
            }

            delta_x_seq[s] = delta_x;
            delta_h = delta_h_prev;
        }

        return delta_x_seq;
    }

    std::vector<float> backward(const std::vector<float>& delta_h, float lr) {
        if (history_.empty()) return std::vector<float>(input_dim, 0.f);
        auto seq = backward_through_time({delta_h}, lr, 1);
        if (seq.empty()) return std::vector<float>(input_dim, 0.f);
        return seq[0];
    }

    void clear_history() { history_.clear(); }
    void reset_state() {
        std::fill(h.begin(), h.end(), 0.f);
        std::fill(c.begin(), c.end(), 0.f);
        history_.clear();
    }

    void save(std::ofstream& f) const {
        f.write((const char*)&input_dim, sizeof(int));
        f.write((const char*)&hidden_dim, sizeof(int));
        f.write((const char*)&k_active, sizeof(int));
        router.save(f);
        
        auto wv = [&](const std::vector<float> &v) {
            size_t n = v.size();
            f.write((const char *)&n, sizeof(size_t));
            f.write((const char *)v.data(), (std::streamsize)(n * sizeof(float)));
        };
        wv(Wh);
        wv(Wx);
        wv(b);
        wv(h);
        wv(c);
    }

    static SparseLSTMLayer load(std::ifstream& f) {
        SparseLSTMLayer l;
        f.read((char*)&l.input_dim, sizeof(int));
        f.read((char*)&l.hidden_dim, sizeof(int));
        f.read((char*)&l.k_active, sizeof(int));
        l.router = LSHRouter::load(f);
        
        auto rv = [&](std::vector<float> &v) {
            size_t n;
            f.read((char *)&n, sizeof(size_t));
            v.resize(n);
            f.read((char *)v.data(), (std::streamsize)(n * sizeof(float)));
        };
        rv(l.Wh);
        rv(l.Wx);
        rv(l.b);
        rv(l.h);
        rv(l.c);
        return l;
    }
};

} // namespace brain2
