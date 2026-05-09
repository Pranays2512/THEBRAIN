#pragma once
/*
 * predictor.hpp — LSTM Prediction Engine, Brain v2
 *
 * BUG FIXED: previous version only updated out_.W and out_.b in step().
 * The two LSTM layers had RANDOM weights for the entire training run.
 * The output layer mapped random LSTM features → targets → collapsed to mean.
 *
 * FIX: added 1-step Truncated BPTT through both LSTM layers.
 *   - LSTMLayer stores last forward pass state (input x, prev h/c, post-activation gates)
 *   - LSTMLayer::backward(delta_h, lr) computes gradients and updates Wh, Wx, b
 *   - Predictor::step() calls backward on lstm2 then lstm1 after output layer update
 *
 * Gradient flow:
 *   loss ← MSE(predicted, actual)
 *   delta_out ← d(loss)/d(pred)                  [sigmoid MSE gradient]
 *   delta_h2  ← W_out^T * delta_out              [backprop through output projection]
 *   delta_x2  ← lstm2_.backward(delta_h2, lr)   [update LSTM2 Wh/Wx/b, return delta_x]
 *   (delta_x2 is the gradient w.r.t. LSTM2's input, which is lstm1's output h1)
 *   lstm1_.backward(delta_x2, lr)                [update LSTM1 Wh/Wx/b]
 *
 * Truncated: no gradient flows back in time (no delta_h_prev, no delta_c_prev).
 * This is sufficient for learning 2–5 step sequence patterns.
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

// ── LSTM helpers ─────────────────────────────────────────────────────

inline void sigmoid_vec(float* v, int n) noexcept {
    for (int i = 0; i < n; i++) v[i] = 1.f / (1.f + std::exp(-v[i]));
}
inline void tanh_vec(float* v, int n) noexcept {
    for (int i = 0; i < n; i++) v[i] = std::tanh(v[i]);
}
inline float l2(const float* a, const float* b, int n) noexcept {
    float s = 0.f;
    for (int i = 0; i < n; i++) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s / n);
}
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
// in-place gradient clipping by global norm
static void clip_inplace(std::vector<float>& g, float max_norm = 5.f) {
    float norm = 0.f;
    for (auto x : g) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (auto& x : g) x *= scale;
    }
}

// ── LSTM Layer with 1-step TBPTT ─────────────────────────────────────

struct LSTMLayer {
    int input_dim, hidden_dim;

    // Weights: Wh[4H×H], Wx[4H×I], b[4H]
    std::vector<float> Wh, Wx, b;
    // Hidden/cell state (updated in-place during forward)
    std::vector<float> h, c;

    // ── Stored for backward pass ─────────────────────────────────────
    std::vector<float> fwd_x;       // input to this layer
    std::vector<float> fwd_h_prev;  // h before this step
    std::vector<float> fwd_c_prev;  // c before this step (for f-gate grad)
    // post-activation gates: [f | i | g | o], each H elements
    std::vector<float> fwd_gates;
    bool               has_fwd = false;

    LSTMLayer() = default;

    LSTMLayer(int input_dim, int hidden_dim, std::mt19937& rng)
        : input_dim(input_dim), hidden_dim(hidden_dim)
    {
        int H = hidden_dim, I = input_dim;
        float scale_h = std::sqrt(1.f / H);
        float scale_x = std::sqrt(2.f / (I + H));
        std::normal_distribution<float> dh(0.f, scale_h);
        std::normal_distribution<float> dx(0.f, scale_x);

        Wh.resize((size_t)4*H * H); for (auto& w : Wh) w = dh(rng);
        Wx.resize((size_t)4*H * I); for (auto& w : Wx) w = dx(rng);
        b.resize(4*H, 0.f);
        for (int i = 0; i < H; i++) b[i] = 1.f;  // forget gate bias

        h.resize(H, 0.f);
        c.resize(H, 0.f);
        fwd_x.resize(I, 0.f);
        fwd_h_prev.resize(H, 0.f);
        fwd_c_prev.resize(H, 0.f);
        fwd_gates.resize(4*H, 0.f);
    }

    // Forward pass — returns new hidden state h.
    // Saves input, prev h/c, and post-activation gates for backward.
    std::vector<float> forward(const std::vector<float>& x) {
        int H = hidden_dim;
        // Save state before update
        fwd_x      = x;
        fwd_h_prev = h;
        fwd_c_prev = c;

        std::vector<float> gates(4*H, 0.f);
        std::copy(b.begin(), b.end(), gates.begin());
        matvec(Wx.data(), x.data(), gates.data(), 4*H, input_dim);
        matvec(Wh.data(), h.data(), gates.data(), 4*H, H);

        float* f  = gates.data();
        float* i_ = gates.data() + H;
        float* g  = gates.data() + 2*H;
        float* o  = gates.data() + 3*H;
        sigmoid_vec(f,  H);
        sigmoid_vec(i_, H);
        tanh_vec   (g,  H);
        sigmoid_vec(o,  H);

        // Save post-activation gates
        fwd_gates = gates;
        has_fwd   = true;

        for (int k = 0; k < H; k++) c[k] = f[k]*c[k] + i_[k]*g[k];
        for (int k = 0; k < H; k++) h[k] = o[k] * std::tanh(c[k]);
        return h;
    }

    // 1-step TBPTT backward.
    // delta_h: gradient w.r.t. this layer's output h.
    // Returns delta_x: gradient w.r.t. this layer's input x
    //   (propagate to the previous layer or discard for layer 1).
    std::vector<float> backward(const std::vector<float>& delta_h, float lr) {
        if (!has_fwd) return std::vector<float>(input_dim, 0.f);
        int H = hidden_dim, I = input_dim;

        const float* f  = fwd_gates.data();
        const float* i_ = fwd_gates.data() + H;
        const float* g  = fwd_gates.data() + 2*H;
        const float* o  = fwd_gates.data() + 3*H;

        // tanh(c) — current cell state: h = o * tanh(c) => tanh(c) = h/o.
        // More stable: compute from stored c directly.
        // But c has already been updated in forward. We need tanh(c_new).
        // h[k] = o[k] * tanh(c[k])  =>  tanh_c[k] = h[k] / o[k] (numerically risky).
        // Instead recompute: tanh_c[k] = tanh(c[k]).
        // c[k] is the current (post-forward) value. This is what we need.
        std::vector<float> tanh_c(H);
        for (int k = 0; k < H; k++) tanh_c[k] = std::tanh(c[k]);

        // Gradients through h = o * tanh(c)
        std::vector<float> dO(H), dC(H), dI(H), dG(H), dF(H);
        for (int k = 0; k < H; k++) {
            dO[k] = delta_h[k] * tanh_c[k];
            float d_tanh = delta_h[k] * o[k] * (1.f - tanh_c[k]*tanh_c[k]);
            dC[k] = d_tanh;
            // c = f*c_prev + i*g
            dI[k] = dC[k] * g[k];
            dG[k] = dC[k] * i_[k];
            dF[k] = dC[k] * fwd_c_prev[k];
        }

        // Pre-activation deltas (gate derivative: sigmoid'(x)=s(1-s), tanh'(x)=1-t^2)
        std::vector<float> dpre(4*H);
        for (int k = 0; k < H; k++) {
            dpre[k]       = dF[k] * f[k]  * (1.f - f[k]);   // forget (sigmoid)
            dpre[H   + k] = dI[k] * i_[k] * (1.f - i_[k]);  // input  (sigmoid)
            dpre[2*H + k] = dG[k] * (1.f - g[k]*g[k]);       // cell   (tanh)
            dpre[3*H + k] = dO[k] * o[k]  * (1.f - o[k]);   // output (sigmoid)
        }

        // Clip pre-activation gradients to stabilise training
        clip_inplace(dpre, 5.f);

        // Update Wx  [4H×I] -= lr * dpre ⊗ fwd_x
        for (int i = 0; i < 4*H; i++)
            for (int j = 0; j < I; j++)
                Wx[(size_t)i * I + j] -= lr * dpre[i] * fwd_x[j];

        // Update Wh  [4H×H] -= lr * dpre ⊗ fwd_h_prev
        for (int i = 0; i < 4*H; i++)
            for (int j = 0; j < H; j++)
                Wh[(size_t)i * H + j] -= lr * dpre[i] * fwd_h_prev[j];

        // Update b   [4H] -= lr * dpre
        for (int i = 0; i < 4*H; i++)
            b[i] -= lr * dpre[i];

        // Compute delta_x = Wx^T * dpre  (gradient w.r.t. input)
        std::vector<float> delta_x(I, 0.f);
        for (int j = 0; j < I; j++)
            for (int i = 0; i < 4*H; i++)
                delta_x[j] += Wx[(size_t)i * I + j] * dpre[i];

        return delta_x;
    }

    void reset_state() {
        std::fill(h.begin(), h.end(), 0.f);
        std::fill(c.begin(), c.end(), 0.f);
        has_fwd = false;
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
        l.fwd_x.resize(I, 0.f);
        l.fwd_h_prev.resize(H, 0.f);
        l.fwd_c_prev.resize(H, 0.f);
        l.fwd_gates.resize(4*H, 0.f);
        l.has_fwd = false;
        return l;
    }
};

// ── Output Projection ────────────────────────────────────────────────

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
        sigmoid_vec(out.data(), output_dim);
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

// ── Predictor — 2-layer LSTM with full TBPTT ─────────────────────────

class Predictor {
public:
    int input_dim, hidden_dim, output_dim;

private:
    LSTMLayer   lstm1_, lstm2_;
    OutputLayer out_;
    float       lr_;
    float       last_error_;
    bool        offline_;
    std::unique_ptr<std::mutex> mtx_;

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

    void set_offline(bool offline) { offline_ = offline; }
    bool is_offline() const { return offline_; }

    std::vector<float> step(const std::vector<float>& input,
                             const std::vector<float>* actual = nullptr) {
        std::lock_guard<std::mutex> lock(*mtx_);

        // ── Forward ──────────────────────────────────────────────────
        auto h1   = lstm1_.forward(input);
        auto h2   = lstm2_.forward(h1);
        auto pred = out_.forward(h2);

        if (!offline_ && actual != nullptr) {
            last_error_ = l2(pred.data(), actual->data(), output_dim);
            int H = hidden_dim, D = output_dim;

            // ── Output layer gradient ─────────────────────────────────
            std::vector<float> delta_out(D);
            for (int i = 0; i < D; i++) {
                float p = pred[i];
                delta_out[i] = 2.f * (p - (*actual)[i]) * p * (1.f - p);
            }
            // Clip output gradient
            clip_inplace(delta_out, 5.f);

            // Update output layer weights
            for (int i = 0; i < D; i++)
                for (int j = 0; j < H; j++)
                    out_.W[(size_t)i * H + j] -= lr_ * delta_out[i] * h2[j];
            for (int i = 0; i < D; i++)
                out_.b[i] -= lr_ * delta_out[i];

            // ── Backprop through LSTM2 (FIXED: was missing entirely) ──
            // delta_h2 = W_out^T * delta_out
            std::vector<float> delta_h2(H, 0.f);
            for (int j = 0; j < H; j++)
                for (int i = 0; i < D; i++)
                    delta_h2[j] += out_.W[(size_t)i * H + j] * delta_out[i];

            // 1-step TBPTT through LSTM2; returns delta w.r.t. LSTM2's input (= h1)
            auto delta_h1 = lstm2_.backward(delta_h2, lr_);

            // ── Backprop through LSTM1 (FIXED: was missing entirely) ──
            lstm1_.backward(delta_h1, lr_);
        }

        return pred;
    }

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