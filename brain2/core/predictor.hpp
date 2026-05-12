#pragma once
/*
 * predictor.hpp — LSTM Prediction Engine, Brain v2
 *
 * FIXES APPLIED:
 *   v1 — LSTM weights were never trained (only out_.W updated). Fixed with
 * 1-step TBPTT. v2 — delta_x computed after Wx update (used wrong weights).
 * Fixed ordering. v3 — 1-step TBPTT cannot propagate gradient far enough for
 * arithmetic or ConceptNet triples. Added: • ForwardSnapshot history buffer in
 * LSTMLayer (up to MAX_HISTORY steps) • backward_through_time(delta_h, lr,
 * n_steps) — true N-step TBPTT • Predictor::train_sequence(inputs, target) —
 * answer-only loss + full BPTT
 *
 * WHY ANSWER-ONLY LOSS:
 *   Training "2 + 3 = ?" with next-token loss gives conflicting targets:
 *     input=2  → predict +     (right)
 *     input=+  → predict 3     (right)
 *     input==  → predict 5     (right, but gradient only reaches 1 step back)
 *   The gradient for "predict 5" never reaches the weights that saw "2" and
 * "3". With train_sequence([2,+,3,=], target=5): — forward: LSTM sees [2, +, 3,
 * =] in order — loss: only at the final output (should be 5) — BPTT: gradient
 * flows back through =, 3, +, 2 — all weights get signal
 *
 * PYBIND11: add to your bindings file:
 *   .def("train_sequence",
 *        [](Predictor& p,
 *           const std::vector<std::vector<float>>& inputs,
 *           const std::vector<float>& target,
 *           int n_bptt) { return p.train_sequence(inputs, target, n_bptt); },
 *        py::arg("inputs"), py::arg("target"), py::arg("n_bptt") = -1)
 *
 * OpenMP: compile with -fopenmp (-Xpreprocessor -fopenmp -lomp on macOS).
 *   Pragmas are silently ignored without OpenMP — code remains correct.
 */

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_math.cuh"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace brain2 {

// ── LSTM helpers ──────────────────────────────────────────────────────

inline void sigmoid_vec(float *v, int n) noexcept {
  for (int i = 0; i < n; i++)
    v[i] = 1.f / (1.f + std::exp(-v[i]));
}
inline void tanh_vec(float *v, int n) noexcept {
  for (int i = 0; i < n; i++)
    v[i] = std::tanh(v[i]);
}
inline float l2(const float *a, const float *b, int n) noexcept {
  float s = 0.f;
  for (int i = 0; i < n; i++) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s / n);
}

// out[m] += W[m×n] * x[n] — parallelized over rows (each row → distinct out[i])
inline void matvec(const float *__restrict__ W, const float *__restrict__ x,
                   float *__restrict__ out, int m, int n) noexcept {
#ifdef USE_CUDA
    cuda_matvec_add(W, x, out, m, n);
    cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; i++) {
    float s = 0.f;
    const float *row = W + (size_t)i * n;
    for (int j = 0; j < n; j++)
      s += row[j] * x[j];
    out[i] += s;
  }
#endif
}

static void clip_inplace(std::vector<float> &g, float max_norm = 5.f) {
  float norm = 0.f;
  for (auto x : g)
    norm += x * x;
  norm = std::sqrt(norm);
  if (norm > max_norm) {
    float scale = max_norm / norm;
    for (auto &x : g)
      x *= scale;
  }
}

// ── Forward snapshot (stored for N-step BPTT) ─────────────────────────

struct ForwardSnapshot {
  std::vector<float> x;      // layer input at this step
  std::vector<float> h_prev; // h before this step
  std::vector<float> c_prev; // c before this step
  std::vector<float> gates;  // post-activation [f | i | g | o], each H elems
  std::vector<float> c_new;  // c after this step (for tanh(c) in backward)
};

// ── LSTM Layer ─────────────────────────────────────────────────────────

struct LSTMLayer {
  static constexpr int MAX_HISTORY = 24; // max BPTT depth

  int input_dim, hidden_dim;

  DeviceVector<float> Wh, Wx, b; // Wh[4H×H], Wx[4H×I], b[4H]
  DeviceVector<float> h, c;      // current hidden + cell state

  // 1-step BPTT state (used by step() backward)
  std::vector<float> fwd_x;
  std::vector<float> fwd_h_prev;
  std::vector<float> fwd_c_prev;
  std::vector<float> fwd_gates;
  bool has_fwd = false;

  // N-step BPTT history (used by backward_through_time)
  std::deque<ForwardSnapshot> history_;

  LSTMLayer() = default;

  LSTMLayer(int input_dim, int hidden_dim, std::mt19937 &rng)
      : input_dim(input_dim), hidden_dim(hidden_dim) {
    int H = hidden_dim, I = input_dim;
    float scale_h = std::sqrt(1.f / H);
    float scale_x = std::sqrt(2.f / (I + H));
    std::normal_distribution<float> dh(0.f, scale_h);
    std::normal_distribution<float> dx(0.f, scale_x);

    Wh.resize((size_t)4 * H * H);
    for (auto &w : Wh)
      w = dh(rng);
    Wx.resize((size_t)4 * H * I);
    for (auto &w : Wx)
      w = dx(rng);
    b.resize(4 * H, 0.f);
    for (int i = 0; i < H; i++)
      b[i] = 1.f; // forget gate bias

    h.resize(H, 0.f);
    c.resize(H, 0.f);
    fwd_x.resize(I, 0.f);
    fwd_h_prev.resize(H, 0.f);
    fwd_c_prev.resize(H, 0.f);
    fwd_gates.resize(4 * H, 0.f);
  }

  // Forward — saves snapshot to history_ for N-step BPTT
  // and also saves to fwd_* for 1-step BPTT (backward())
  std::vector<float> forward(const std::vector<float> &x,
                             bool record_history = false) {
    int H = hidden_dim;

    // Save pre-step state
    fwd_x = x;
    fwd_h_prev = h;
    fwd_c_prev = c;

    std::vector<float> gates(4 * H, 0.f);
    std::copy(b.begin(), b.end(), gates.begin());
    matvec(Wx.data(), x.data(), gates.data(), 4 * H, input_dim);
    matvec(Wh.data(), h.data(), gates.data(), 4 * H, H);

    float *f = gates.data();
    float *i_ = gates.data() + H;
    float *g = gates.data() + 2 * H;
    float *o = gates.data() + 3 * H;
    sigmoid_vec(f, H);
    sigmoid_vec(i_, H);
    tanh_vec(g, H);
    sigmoid_vec(o, H);

    fwd_gates = gates;
    has_fwd = true;

    for (int k = 0; k < H; k++)
      c[k] = f[k] * c[k] + i_[k] * g[k];
    for (int k = 0; k < H; k++)
      h[k] = o[k] * std::tanh(c[k]);

    if (record_history) {
      ForwardSnapshot snap;
      snap.x = x;
      snap.h_prev = fwd_h_prev;
      snap.c_prev = fwd_c_prev;
      snap.gates = gates;
      snap.c_new = c; // c is now c_new
      history_.push_back(std::move(snap));
      if ((int)history_.size() > MAX_HISTORY)
        history_.pop_front();
    }

    return h;
  }

  // 1-step BPTT backward (used by Predictor::step())
  // Computes delta_x from Wx_OLD before updating Wx.
  std::vector<float> backward(const std::vector<float> &delta_h, float lr) {
    if (!has_fwd)
      return std::vector<float>(input_dim, 0.f);
    int H = hidden_dim, I = input_dim;

    const float *f = fwd_gates.data();
    const float *i_ = fwd_gates.data() + H;
    const float *g = fwd_gates.data() + 2 * H;
    const float *o = fwd_gates.data() + 3 * H;

    std::vector<float> tanh_c(H);
    for (int k = 0; k < H; k++)
      tanh_c[k] = std::tanh(c[k]);

    std::vector<float> dO(H), dC(H), dI(H), dG(H), dF(H);
    for (int k = 0; k < H; k++) {
      dO[k] = delta_h[k] * tanh_c[k];
      float d_tanh = delta_h[k] * o[k] * (1.f - tanh_c[k] * tanh_c[k]);
      dC[k] = d_tanh;
      dI[k] = dC[k] * g[k];
      dG[k] = dC[k] * i_[k];
      dF[k] = dC[k] * fwd_c_prev[k];
    }

    std::vector<float> dpre(4 * H);
    for (int k = 0; k < H; k++) {
      dpre[k] = dF[k] * f[k] * (1.f - f[k]);
      dpre[H + k] = dI[k] * i_[k] * (1.f - i_[k]);
      dpre[2 * H + k] = dG[k] * (1.f - g[k] * g[k]);
      dpre[3 * H + k] = dO[k] * o[k] * (1.f - o[k]);
    }
    clip_inplace(dpre, 5.f);

    // STEP 1: delta_x from Wx_OLD (before Wx update)
    std::vector<float> delta_x(I, 0.f);
#ifdef USE_CUDA
    // Since this is W^T * dpre, we don't have a specific kernel for transposed matvec,
    // so we'll fallback to CPU for delta_x and then use GPU for weight updates.
    cuda_device_synchronize();
#endif
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < I; j++) {
      float s = 0.f;
      for (int i = 0; i < 4 * H; i++)
        s += Wx[(size_t)i * I + j] * dpre[i];
      delta_x[j] = s;
    }

    // STEP 2: update Wx
#ifdef USE_CUDA
    cuda_matvec_sub(Wx.data(), fwd_x.data(), dpre.data(), 4 * H, I);
    cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < 4 * H; i++) {
      float dp = lr * dpre[i];
      float *row = Wx.data() + (size_t)i * I;
      const float *xp = fwd_x.data();
      for (int j = 0; j < I; j++)
        row[j] -= dp * xp[j];
    }
#endif

    // STEP 3: update Wh
#ifdef USE_CUDA
    cuda_matvec_sub(Wh.data(), fwd_h_prev.data(), dpre.data(), 4 * H, H);
    cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < 4 * H; i++) {
      float dp = lr * dpre[i];
      float *row = Wh.data() + (size_t)i * H;
      const float *hp = fwd_h_prev.data();
      for (int j = 0; j < H; j++)
        row[j] -= dp * hp[j];
    }
#endif


    // STEP 4: update b
    for (int i = 0; i < 4 * H; i++)
      b[i] -= lr * dpre[i];

    return delta_x;
  }

  // N-step BPTT backward through history_.
  // delta_h_last: gradient at the last timestep's output h.
  // n_steps: how many steps to unroll (-1 = full history).
  // Returns: delta_x at the EARLIEST step (for propagating to layer below).
  //
  // Gradient at each step flows:
  //   delta_h → gates → delta_h_prev (to previous timestep's h)
  //             gates → delta_x      (to this timestep's input x)
  // Wx/Wh/b updated at each step using OLD weights (delta_x/delta_h_prev
  // computed before weight update at that step).
  std::vector<float>
  backward_through_time(const std::vector<float> &delta_h_last, float lr,
                        int n_steps = -1) {
    int H = hidden_dim, I = input_dim;
    int hist = (int)history_.size();
    int steps = (n_steps < 0) ? hist : std::min(n_steps, hist);
    if (steps == 0)
      return std::vector<float>(I, 0.f);

    std::vector<float> delta_h = delta_h_last;
    std::vector<float> delta_x_earliest(I, 0.f);

    // Unroll from most recent to oldest snapshot
    for (int s = steps - 1; s >= 0; s--) {
      const ForwardSnapshot &snap = history_[hist - steps + s];

      const float *f = snap.gates.data();
      const float *i_ = snap.gates.data() + H;
      const float *g = snap.gates.data() + 2 * H;
      const float *o = snap.gates.data() + 3 * H;

      // tanh(c_new) stored in snapshot
      std::vector<float> tanh_c(H);
      for (int k = 0; k < H; k++)
        tanh_c[k] = std::tanh(snap.c_new[k]);

      // Gate gradients
      std::vector<float> dO(H), dC(H), dI(H), dG(H), dF(H);
      for (int k = 0; k < H; k++) {
        dO[k] = delta_h[k] * tanh_c[k];
        float d_tanh = delta_h[k] * o[k] * (1.f - tanh_c[k] * tanh_c[k]);
        dC[k] = d_tanh;
        dI[k] = dC[k] * g[k];
        dG[k] = dC[k] * i_[k];
        dF[k] = dC[k] * snap.c_prev[k];
      }

      std::vector<float> dpre(4 * H);
      for (int k = 0; k < H; k++) {
        dpre[k] = dF[k] * f[k] * (1.f - f[k]);
        dpre[H + k] = dI[k] * i_[k] * (1.f - i_[k]);
        dpre[2 * H + k] = dG[k] * (1.f - g[k] * g[k]);
        dpre[3 * H + k] = dO[k] * o[k] * (1.f - o[k]);
      }
      clip_inplace(dpre, 5.f);

      // delta_x from Wx_OLD (before Wx update) — propagates to layer below
      std::vector<float> delta_x(I, 0.f);
#ifdef USE_CUDA
      cuda_device_synchronize();
#endif
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int j = 0; j < I; j++) {
        float acc = 0.f;
        for (int ii = 0; ii < 4 * H; ii++)
          acc += Wx[(size_t)ii * I + j] * dpre[ii];
        delta_x[j] = acc;
      }

      // delta_h_prev from Wh_OLD — gradient flowing back in time
      std::vector<float> delta_h_prev(H, 0.f);
      for (int j = 0; j < H; j++) {
        float acc = 0.f;
        for (int ii = 0; ii < 4 * H; ii++)
          acc += Wh[(size_t)ii * H + j] * dpre[ii];
        delta_h_prev[j] = acc;
      }

      // Update Wx (after delta_x computed)
#ifdef USE_CUDA
      cuda_matvec_sub(Wx.data(), snap.x.data(), dpre.data(), 4 * H, I);
      cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int ii = 0; ii < 4 * H; ii++) {
        float dp = lr * dpre[ii];
        float *row = Wx.data() + (size_t)ii * I;
        const float *xp = snap.x.data();
        for (int j = 0; j < I; j++)
          row[j] -= dp * xp[j];
      }
#endif

      // Update Wh (after delta_h_prev computed)
#ifdef USE_CUDA
      cuda_matvec_sub(Wh.data(), snap.h_prev.data(), dpre.data(), 4 * H, H);
      cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int ii = 0; ii < 4 * H; ii++) {
        float dp = lr * dpre[ii];
        float *row = Wh.data() + (size_t)ii * H;
        const float *hp = snap.h_prev.data();
        for (int j = 0; j < H; j++)
          row[j] -= dp * hp[j];
      }
#endif

      // Update b
      for (int ii = 0; ii < 4 * H; ii++)
        b[ii] -= lr * dpre[ii];

      // Earliest step gives delta_x for the layer below this one
      if (s == 0)
        delta_x_earliest = delta_x;

      // Propagate gradient back one more timestep
      delta_h = delta_h_prev;
    }

    return delta_x_earliest;
  }

  void clear_history() { history_.clear(); }

  void reset_state() {
    std::fill(h.begin(), h.end(), 0.f);
    std::fill(c.begin(), c.end(), 0.f);
    has_fwd = false;
    history_.clear();
  }

  void save(std::ofstream &f) const {
    f.write((const char *)&input_dim, sizeof(int));
    f.write((const char *)&hidden_dim, sizeof(int));
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

  static LSTMLayer load(std::ifstream &f) {
    LSTMLayer l;
    f.read((char *)&l.input_dim, sizeof(int));
    f.read((char *)&l.hidden_dim, sizeof(int));
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
    int H = l.hidden_dim, I = l.input_dim;
    l.fwd_x.resize(I, 0.f);
    l.fwd_h_prev.resize(H, 0.f);
    l.fwd_c_prev.resize(H, 0.f);
    l.fwd_gates.resize(4 * H, 0.f);
    l.has_fwd = false;
    return l;
  }
};

// ── Output Projection ─────────────────────────────────────────────────

struct OutputLayer {
  int input_dim, output_dim;
  DeviceVector<float> W, b;

  OutputLayer() = default;
  OutputLayer(int input_dim, int output_dim, std::mt19937 &rng)
      : input_dim(input_dim), output_dim(output_dim) {
    float scale = std::sqrt(2.f / input_dim);
    std::normal_distribution<float> d(0.f, scale);
    W.resize((size_t)output_dim * input_dim);
    for (auto &w : W)
      w = d(rng);
    b.resize(output_dim, 0.f);
  }

  std::vector<float> forward(const std::vector<float> &x) const {
    std::vector<float> out(b.begin(), b.end());
    matvec(W.data(), x.data(), out.data(), output_dim, input_dim);
    sigmoid_vec(out.data(), output_dim);
    return out;
  }

  void save(std::ofstream &f) const {
    f.write((const char *)&input_dim, sizeof(int));
    f.write((const char *)&output_dim, sizeof(int));
    size_t wn = W.size(), bn = b.size();
    f.write((const char *)&wn, sizeof(size_t));
    f.write((const char *)W.data(), (std::streamsize)(wn * sizeof(float)));
    f.write((const char *)&bn, sizeof(size_t));
    f.write((const char *)b.data(), (std::streamsize)(bn * sizeof(float)));
  }

  static OutputLayer load(std::ifstream &f) {
    OutputLayer l;
    f.read((char *)&l.input_dim, sizeof(int));
    f.read((char *)&l.output_dim, sizeof(int));
    auto rv = [&](std::vector<float> &v) {
      size_t n;
      f.read((char *)&n, sizeof(size_t));
      v.resize(n);
      f.read((char *)v.data(), (std::streamsize)(n * sizeof(float)));
    };
    rv(l.W);
    rv(l.b);
    return l;
  }
};

// ── Predictor ─────────────────────────────────────────────────────────

class Predictor {
public:
  int input_dim, hidden_dim, output_dim;

private:
  LSTMLayer lstm1_, lstm2_;
  OutputLayer out_;
  float lr_;
  float last_error_;
  bool offline_;
  std::unique_ptr<std::mutex> mtx_;

public:
  Predictor()
      : input_dim(0), hidden_dim(0), output_dim(0), lr_(0), last_error_(0),
        offline_(false), mtx_(std::make_unique<std::mutex>()) {}

  Predictor(int input_dim, int hidden_dim = 256, float lr = 0.001f,
            unsigned seed = 42)
      : input_dim(input_dim), hidden_dim(hidden_dim), output_dim(input_dim),
        lr_(lr), last_error_(0.f), offline_(false),
        mtx_(std::make_unique<std::mutex>()) {
    std::mt19937 rng(seed);
    lstm1_ = LSTMLayer(input_dim, hidden_dim, rng);
    lstm2_ = LSTMLayer(hidden_dim, hidden_dim, rng);
    out_ = OutputLayer(hidden_dim, output_dim, rng);
  }

  Predictor(Predictor &&) = default;
  Predictor &operator=(Predictor &&) = default;
  Predictor(const Predictor &) = delete;
  Predictor &operator=(const Predictor &) = delete;

  void set_offline(bool offline) { offline_ = offline; }
  bool is_offline() const { return offline_; }

  // ── Online step (1-step BPTT) ─────────────────────────────────────
  // Used by brain.perceive() and Phase 3 curiosity.
  // Does NOT record to history_ (history_ is only for train_sequence).
  std::vector<float> step(const std::vector<float> &input,
                          const std::vector<float> *actual = nullptr) {
    std::lock_guard<std::mutex> lock(*mtx_);

    auto h1 = lstm1_.forward(input, /*record_history=*/false);
    auto h2 = lstm2_.forward(h1, /*record_history=*/false);
    auto pred = out_.forward(h2);

    if (!offline_ && actual != nullptr) {
      last_error_ = l2(pred.data(), actual->data(), output_dim);
      int H = hidden_dim, D = output_dim;

      std::vector<float> delta_out(D);
      for (int i = 0; i < D; i++) {
        float p = pred[i];
        delta_out[i] = 2.f * (p - (*actual)[i]) * p * (1.f - p);
      }
      clip_inplace(delta_out, 5.f);

      // delta_h2 from W_out_OLD (before W_out update)
      std::vector<float> delta_h2(H, 0.f);
      for (int j = 0; j < H; j++) {
        float s = 0.f;
        for (int i = 0; i < D; i++)
          s += out_.W[(size_t)i * H + j] * delta_out[i];
        delta_h2[j] = s;
      }

      // Update output layer
#ifdef USE_CUDA
      cuda_matvec_sub(out_.W.data(), h2.data(), delta_out.data(), D, H);
      cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < D; i++) {
        float dp = lr_ * delta_out[i];
        float *row = out_.W.data() + (size_t)i * H;
        const float *h2p = h2.data();
        for (int j = 0; j < H; j++)
          row[j] -= dp * h2p[j];
      }
#endif
      for (int i = 0; i < D; i++)
        out_.b[i] -= lr_ * delta_out[i];

      // 1-step TBPTT through LSTM2 then LSTM1
      auto delta_h1 = lstm2_.backward(delta_h2, lr_);
      lstm1_.backward(delta_h1, lr_);
    }

    return pred;
  }

  // ── Sequence training (N-step BPTT, answer-only loss) ─────────────
  //
  // HOW IT WORKS:
  //   1. Forward: feed all `inputs` through LSTM, recording snapshots
  //   2. Loss: compute MSE only at the final output against `target`
  //      (no loss at intermediate steps — answer-only)
  //   3. Backward: full BPTT from loss through all recorded steps
  //      (n_bptt = -1 means unroll through full sequence)
  //
  // TRAINING EXAMPLES:
  //   Math   "2 + 3 = 5": inputs=[act(2),act(+),act(3),act(=)], target=act(5)
  //   ConceptNet "dog isa animal": inputs=[act(dog),act(isa)],
  //   target=act(animal)
  //
  // RETURNS: prediction error (L2) at this step.
  //
  // THREAD SAFETY: locked for full duration.
  float train_sequence(const std::vector<std::vector<float>> &inputs,
                       const std::vector<float> &target, int n_bptt = -1) {
    std::lock_guard<std::mutex> lock(*mtx_);

    if (inputs.empty())
      return 0.f;

    // Clear history so only this sequence's snapshots are used
    lstm1_.clear_history();
    lstm2_.clear_history();

    // Forward through all inputs, recording snapshots
    for (const auto &inp : inputs) {
      auto h1 = lstm1_.forward(inp, /*record_history=*/true);
      lstm2_.forward(h1, /*record_history=*/true);
    }

    // Compute output from lstm2's final hidden state
    auto pred = out_.forward(lstm2_.h);

    // Loss: MSE between prediction and target
    float err = l2(pred.data(), target.data(), output_dim);
    last_error_ = err;

    int H = hidden_dim, D = output_dim;

    // Output layer gradient
    std::vector<float> delta_out(D);
    for (int i = 0; i < D; i++) {
      float p = pred[i];
      delta_out[i] = 2.f * (p - target[i]) * p * (1.f - p);
    }
    clip_inplace(delta_out, 5.f);

    // delta_h2 from W_out_OLD (before W_out update)
    std::vector<float> delta_h2(H, 0.f);
    for (int j = 0; j < H; j++) {
      float s = 0.f;
      for (int i = 0; i < D; i++)
        s += out_.W[(size_t)i * H + j] * delta_out[i];
      delta_h2[j] = s;
    }

    // Update output layer
#ifdef USE_CUDA
    cuda_matvec_sub(out_.W.data(), lstm2_.h.data(), delta_out.data(), D, H);
    cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < D; i++) {
      float dp = lr_ * delta_out[i];
      float *row = out_.W.data() + (size_t)i * H;
      const float *h2p = lstm2_.h.data();
      for (int j = 0; j < H; j++)
        row[j] -= dp * h2p[j];
    }
#endif
    for (int i = 0; i < D; i++)
      out_.b[i] -= lr_ * delta_out[i];

    // N-step BPTT through LSTM2, then LSTM1
    int steps = (n_bptt < 0) ? (int)inputs.size()
                             : std::min(n_bptt, (int)inputs.size());
    auto delta_h1 = lstm2_.backward_through_time(delta_h2, lr_, steps);
    lstm1_.backward_through_time(delta_h1, lr_, steps);

    return err;
  }

  void reset() {
    lstm1_.reset_state();
    lstm2_.reset_state();
  }

  float last_error() const noexcept { return last_error_; }
  float lr() const noexcept { return lr_; }
  void set_lr(float v) { lr_ = v; }

  void save(const std::string &path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("Predictor::save: cannot open " + path);
    f.write((const char *)&input_dim, sizeof(int));
    f.write((const char *)&hidden_dim, sizeof(int));
    f.write((const char *)&lr_, sizeof(float));
    lstm1_.save(f);
    lstm2_.save(f);
    out_.save(f);
  }

  static Predictor load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("Predictor::load: cannot open " + path);
    Predictor p;
    f.read((char *)&p.input_dim, sizeof(int));
    f.read((char *)&p.hidden_dim, sizeof(int));
    f.read((char *)&p.lr_, sizeof(float));
    p.output_dim = p.input_dim;
    p.lstm1_ = LSTMLayer::load(f);
    p.lstm2_ = LSTMLayer::load(f);
    p.out_ = OutputLayer::load(f);
    p.mtx_ = std::make_unique<std::mutex>();
    return p;
  }
};

} // namespace brain2