#pragma once
/*
 * predictor.hpp — LSTM Prediction Engine, Component 4 of Brain v2
 *
 * Predicts next SOM activation vector given sequence of past activations.
 * Prediction error is the primary learning signal for the whole brain.
 *
 * Architecture: 2-layer LSTM, input_dim → hidden_dim → hidden_dim → output_dim
 *   input_dim  = SOM n_dims (same vector space as perception)
 *   hidden_dim = 256 (configurable)
 *   output_dim = SOM n_dims (predicts next activation vector)
 *
 * Two modes:
 *   online  — update weights after every step (default)
 *   offline — run prediction without weight update (imagination mode)
 *
 * Prediction error = L2(predicted, actual)
 * High error = surprise = novel = learning signal sent to all components.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <string>

namespace brain2 {

// ── LSTM cell helpers (inline, auto-vectorized) ──────────────────────

inline void sigmoid_vec(float* v, int n) noexcept {
    for (int i = 0; i < n; i++) v[i] = 1.f / (1.f + std::exp(-v[i]));
}
inline void tanh_vec(float* v, int n) noexcept {
    for (int i = 0; i < n; i++) v[i] = std::tanh(v[i]);
}
inline float l2(const float* a, const float* b, int n) noexcept {
    float s = 0.f;
    for (int i = 0; i < n; i++) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s / n);  // normalized
}

// Matrix-vector multiply: out[m] += W[m×n] * x[n]
inline void matvec(const float* __restrict__ W,
                   const float* __restrict__ x,
                   float*       __restrict__ out,
                   int m, int n) noexcept {
    for (int i = 0; i < m; i++) {
        float s = 0.f;
        const float* row = W + (size_t)i * n;
        for (int j = 0; j < n; j++) s += row[j] * x[j];
        out[i] += s;
    }
}

// ── Single LSTM layer ────────────────────────────────────────────────

struct LSTMLayer {
    int input_dim, hidden_dim;

    // Gates: [forget, input, cell, output] each is hidden_dim
    // Weight shapes: Wh [4H × H], Wx [4H × I], b [4H]
    std::vector<float> Wh, Wx, b;
    std::vector<float> h, c;  // hidden state, cell state

    // Gradient accumulators (for BPTT)
    std::vector<float> dWh, dWx, db;

    LSTMLayer() = default;

    LSTMLayer(int input_dim, int hidden_dim, std::mt19937& rng)
        : input_dim(input_dim), hidden_dim(hidden_dim)
    {
        int H = hidden_dim, I = input_dim;
        float scale_h = std::sqrt(1.f / H);
        float scale_x = std::sqrt(2.f / (I + H));

        std::normal_distribution<float> dh(0.f, scale_h);
        std::normal_distribution<float> dx(0.f, scale_x);

        Wh.resize((size_t)4*H * H);  for (auto& w : Wh) w = dh(rng);
        Wx.resize((size_t)4*H * I);  for (auto& w : Wx) w = dx(rng);
        b.resize(4*H, 0.f);
        // Forget gate bias = 1.0 (helps learning long sequences)
        for (int i = 0; i < H; i++) b[i] = 1.f;

        h.resize(H, 0.f);
        c.resize(H, 0.f);
        dWh.resize((size_t)4*H * H, 0.f);
        dWx.resize((size_t)4*H * I, 0.f);
        db.resize(4*H, 0.f);
    }

    // Forward pass — returns new hidden state
    // Stores gate activations in provided buffers for BPTT
    std::vector<float> forward(const std::vector<float>& x,
                                std::vector<float>* gates_out = nullptr) {
        int H = hidden_dim;
        std::vector<float> gates(4*H, 0.f);

        // gates = Wx*x + Wh*h + b
        std::copy(b.begin(), b.end(), gates.begin());
        matvec(Wx.data(), x.data(), gates.data(), 4*H, input_dim);
        matvec(Wh.data(), h.data(), gates.data(), 4*H, H);

        float* f = gates.data();        // forget  [0..H)
        float* i_ = gates.data() + H;  // input   [H..2H)
        float* g = gates.data() + 2*H; // cell    [2H..3H)
        float* o = gates.data() + 3*H; // output  [3H..4H)

        sigmoid_vec(f,  H);
        sigmoid_vec(i_, H);
        tanh_vec(g,     H);
        sigmoid_vec(o,  H);

        // c = f * c_prev + i * g
        for (int k = 0; k < H; k++) c[k] = f[k]*c[k] + i_[k]*g[k];
        // h = o * tanh(c)
        for (int k = 0; k < H; k++) h[k] = o[k] * std::tanh(c[k]);

        if (gates_out) *gates_out = gates;
        return h;
    }

    void reset_state() {
        std::fill(h.begin(), h.end(), 0.f);
        std::fill(c.begin(), c.end(), 0.f);
    }

    void save(std::ofstream& f) const {
        f.write((const char*)&input_dim,  sizeof(int));
        f.write((const char*)&hidden_dim, sizeof(int));
        auto write_vec = [&](const std::vector<float>& v) {
            size_t n = v.size();
            f.write((const char*)&n, sizeof(size_t));
            f.write((const char*)v.data(), (std::streamsize)(n * sizeof(float)));
        };
        write_vec(Wh); write_vec(Wx); write_vec(b);
        write_vec(h);  write_vec(c);
    }

    static LSTMLayer load(std::ifstream& f) {
        LSTMLayer l;
        f.read((char*)&l.input_dim,  sizeof(int));
        f.read((char*)&l.hidden_dim, sizeof(int));
        auto read_vec = [&](std::vector<float>& v) {
            size_t n; f.read((char*)&n, sizeof(size_t));
            v.resize(n);
            f.read((char*)v.data(), (std::streamsize)(n * sizeof(float)));
        };
        read_vec(l.Wh); read_vec(l.Wx); read_vec(l.b);
        read_vec(l.h);  read_vec(l.c);
        int H = l.hidden_dim, I = l.input_dim;
        l.dWh.resize((size_t)4*H * H, 0.f);
        l.dWx.resize((size_t)4*H * I, 0.f);
        l.db.resize(4*H, 0.f);
        return l;
    }
};

// ── Output projection ────────────────────────────────────────────────

struct OutputLayer {
    int input_dim, output_dim;
    std::vector<float> W, b;

    OutputLayer() = default;
    OutputLayer(int input_dim, int output_dim, std::mt19937& rng)
        : input_dim(input_dim), output_dim(output_dim)
    {
        float scale = std::sqrt(2.f / input_dim);
        std::normal_distribution<float> d(0.f, scale);
        W.resize((size_t)output_dim * input_dim);
        for (auto& w : W) w = d(rng);
        b.resize(output_dim, 0.f);
    }

    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> out(b.begin(), b.end());
        matvec(W.data(), x.data(), out.data(), output_dim, input_dim);
        return out;
    }

    void save(std::ofstream& f) const {
        f.write((const char*)&input_dim,  sizeof(int));
        f.write((const char*)&output_dim, sizeof(int));
        size_t wn = W.size(), bn = b.size();
        f.write((const char*)&wn, sizeof(size_t));
        f.write((const char*)W.data(), (std::streamsize)(wn * sizeof(float)));
        f.write((const char*)&bn, sizeof(size_t));
        f.write((const char*)b.data(), (std::streamsize)(bn * sizeof(float)));
    }

    static OutputLayer load(std::ifstream& f) {
        OutputLayer l;
        f.read((char*)&l.input_dim,  sizeof(int));
        f.read((char*)&l.output_dim, sizeof(int));
        auto read_vec = [&](std::vector<float>& v) {
            size_t n; f.read((char*)&n, sizeof(size_t));
            v.resize(n);
            f.read((char*)v.data(), (std::streamsize)(n * sizeof(float)));
        };
        read_vec(l.W); read_vec(l.b);
        return l;
    }
};

// ── Predictor — 2-layer LSTM ─────────────────────────────────────────

class Predictor {
public:
    int input_dim, hidden_dim, output_dim;

private:
    LSTMLayer  lstm1_, lstm2_;
    OutputLayer out_;
    float      lr_;
    float      last_error_;
    bool       offline_;
    std::unique_ptr<std::mutex> mtx_;

    // Simple gradient clipping
    static void clip_grad(std::vector<float>& g, float max_norm = 5.f) {
        float norm = 0.f;
        for (auto x : g) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > max_norm) {
            float scale = max_norm / norm;
            for (auto& x : g) x *= scale;
        }
    }

public:
    Predictor() : input_dim(0), hidden_dim(0), output_dim(0),
                  lr_(0), last_error_(0), offline_(false),
                  mtx_(std::make_unique<std::mutex>()) {}

    Predictor(int input_dim, int hidden_dim = 256,
              float lr = 0.001f, unsigned seed = 42)
        : input_dim(input_dim), hidden_dim(hidden_dim), output_dim(input_dim),
          lr_(lr), last_error_(0.f), offline_(false),
          mtx_(std::make_unique<std::mutex>())
    {
        std::mt19937 rng(seed);
        lstm1_ = LSTMLayer(input_dim,  hidden_dim, rng);
        lstm2_ = LSTMLayer(hidden_dim, hidden_dim, rng);
        out_   = OutputLayer(hidden_dim, output_dim, rng);
    }

    Predictor(Predictor&&)            = default;
    Predictor& operator=(Predictor&&) = default;
    Predictor(const Predictor&)       = delete;
    Predictor& operator=(const Predictor&) = delete;

    // Set offline mode — predictions run but weights don't update (imagination)
    void set_offline(bool offline) { offline_ = offline; }
    bool is_offline() const { return offline_; }

    // Predict next activation given current input
    // If online and actual provided: compute error, update weights (truncated BPTT-1)
    std::vector<float> step(const std::vector<float>& input,
                             const std::vector<float>* actual = nullptr) {
        std::lock_guard<std::mutex> lock(*mtx_);

        // Forward pass through both LSTM layers
        auto h1 = lstm1_.forward(input);
        auto h2 = lstm2_.forward(h1);
        auto pred = out_.forward(h2);

        if (!offline_ && actual != nullptr) {
            // Compute prediction error
            last_error_ = l2(pred.data(), actual->data(), output_dim);

            // Simplified 1-step gradient update (output layer + LSTM2 hidden)
            // Full BPTT is too expensive online; 1-step gives good signal
            int H = hidden_dim, D = output_dim;

            // Output layer gradient: dL/dW = delta * h2^T
            std::vector<float> delta_out(D);
            for (int i = 0; i < D; i++)
                delta_out[i] = 2.f * (pred[i] - (*actual)[i]) / D;

            // Update output weights
            for (int i = 0; i < D; i++)
                for (int j = 0; j < H; j++)
                    out_.W[(size_t)i * H + j] -= lr_ * delta_out[i] * h2[j];
            for (int i = 0; i < D; i++)
                out_.b[i] -= lr_ * delta_out[i];

            // Backprop through output projection to LSTM2 hidden
            std::vector<float> dh2(H, 0.f);
            for (int j = 0; j < H; j++)
                for (int i = 0; i < D; i++)
                    dh2[j] += out_.W[(size_t)i * H + j] * delta_out[i];

            // Update LSTM2 input weights (x = h1) with learning rate scaled by error
            float scale = lr_ * std::min(last_error_, 1.f);
            for (int i = 0; i < 4*H; i++)
                for (int j = 0; j < H; j++)
                    lstm2_.Wx[(size_t)i * H + j] -= scale * dh2[i % H] * h1[j];
        }

        return pred;
    }

    // Reset temporal state (start of new sequence)
    void reset() {
        lstm1_.reset_state();
        lstm2_.reset_state();
    }

    float last_error()    const noexcept { return last_error_; }
    float lr()            const noexcept { return lr_; }
    void  set_lr(float v)              { lr_ = v; }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Predictor::save: cannot open " + path);
        f.write((const char*)&input_dim,  sizeof(int));
        f.write((const char*)&hidden_dim, sizeof(int));
        f.write((const char*)&lr_,        sizeof(float));
        lstm1_.save(f); lstm2_.save(f); out_.save(f);
    }

    static Predictor load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Predictor::load: cannot open " + path);
        Predictor p;
        f.read((char*)&p.input_dim,  sizeof(int));
        f.read((char*)&p.hidden_dim, sizeof(int));
        f.read((char*)&p.lr_,        sizeof(float));
        p.output_dim = p.input_dim;
        p.lstm1_ = LSTMLayer::load(f);
        p.lstm2_ = LSTMLayer::load(f);
        p.out_   = OutputLayer::load(f);
        p.mtx_   = std::make_unique<std::mutex>();
        return p;
    }
};

} // namespace brain2
