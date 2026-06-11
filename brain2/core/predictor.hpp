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

#include "sparse_lstm.hpp"

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
  // Sparse optimization for large n
  std::vector<int> nz_idx;
  nz_idx.reserve(std::min(n, 256));
  for (int j = 0; j < n; j++) {
      if (std::abs(x[j]) > 1e-6f) nz_idx.push_back(j);
  }
  
  for (int i = 0; i < m; i++) {
    float s = 0.f;
    const float *row = W + (size_t)i * n;
    for (int j : nz_idx)
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

// ── Output Projection ─────────────────────────────────────────────────

struct OutputLayer {
  int input_dim;
  int K;
  int target_dim;
  int output_dim;
  DeviceVector<float> W, b;

  OutputLayer() = default;
  OutputLayer(int input_dim, int K, int target_dim, std::mt19937 &rng)
      : input_dim(input_dim), K(K), target_dim(target_dim), output_dim(K * (target_dim + 1)) {
    float scale = std::sqrt(2.f / input_dim);
    std::normal_distribution<float> d(0.f, scale);
    W.resize((size_t)output_dim * input_dim);
    for (auto &w : W)
      w = d(rng);
    b.resize(output_dim, 0.f); 
    // Initialize confidence bias to -2.0 to start with low confidence (~11%)
    for(int k=0; k<K; k++) {
        b[k * (target_dim + 1) + target_dim] = -2.0f;
    }
  }

  std::vector<float> forward(const std::vector<float> &x) const {
    std::vector<float> out(b.begin(), b.end());
    matvec(W.data(), x.data(), out.data(), output_dim, input_dim);
    // Apply sigmoid only to the confidence scores, leave coordinates linear!
    for(int k=0; k<K; k++) {
        int conf_idx = k * (target_dim + 1) + target_dim;
        out[conf_idx] = 1.f / (1.f + std::exp(-out[conf_idx]));
    }
    return out;
  }

  void save(std::ofstream &f) const {
    f.write((const char *)&input_dim, sizeof(int));
    f.write((const char *)&K, sizeof(int));
    f.write((const char *)&target_dim, sizeof(int));
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
    f.read((char *)&l.K, sizeof(int));
    f.read((char *)&l.target_dim, sizeof(int));
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

  void expand_output_dim(int new_target_dim) {
      if (new_target_dim <= target_dim) return;
      int new_out = K * (new_target_dim + 1);
      int H = input_dim;
      std::vector<float> new_W(new_out * H, 0.f);
      std::vector<float> new_b(new_out, 0.f);
      for (int k = 0; k < K; k++) {
          int old_offset = k * (target_dim + 1);
          int new_offset = k * (new_target_dim + 1);
          // Copy coordinate weights
          for(int i=0; i<target_dim; i++) {
              for (int j = 0; j < H; j++) {
                  new_W[(new_offset + i) * H + j] = W[(old_offset + i) * H + j];
              }
              new_b[new_offset + i] = b[old_offset + i];
          }
          // Copy confidence weights
          for(int j=0; j<H; j++) {
              new_W[(new_offset + new_target_dim) * H + j] = W[(old_offset + target_dim) * H + j];
          }
          new_b[new_offset + new_target_dim] = b[old_offset + target_dim];
      }
      W = std::move(new_W);
      b = std::move(new_b);
      target_dim = new_target_dim;
      output_dim = new_out;
  }
};

// ── Predictor ─────────────────────────────────────────────────────────

class Predictor {
public:
  int input_dim, hidden_dim, target_dim, K;

private:
  SparseLSTMLayer lstm1_, lstm2_;
  OutputLayer out_;
  float lr_;
  float last_error_;
  bool offline_;
  std::unique_ptr<std::mutex> mtx_;
  std::deque<std::vector<float>> h2_history_;
  static constexpr int MAX_ATTN_HISTORY = 24;

public:
  Predictor()
      : input_dim(0), hidden_dim(0), target_dim(0), K(0), lr_(0), last_error_(0),
        offline_(false), mtx_(std::make_unique<std::mutex>()) {}

  Predictor(int input_dim, int hidden_dim = 256, int target_dim = 128, int K = 5, float lr = 0.001f,
            unsigned seed = 42)
      : input_dim(input_dim), hidden_dim(hidden_dim), target_dim(target_dim), K(K),
        lr_(lr), last_error_(0.f), offline_(false),
        mtx_(std::make_unique<std::mutex>()) {
    std::mt19937 rng(seed);
    // k_active = 64 out of 512 neurons (8x sparsity)
    lstm1_ = SparseLSTMLayer(input_dim, hidden_dim, 64, rng);
    lstm2_ = SparseLSTMLayer(hidden_dim, hidden_dim, 64, rng);
    out_ = OutputLayer(hidden_dim, K, target_dim, rng);
  }

  Predictor(Predictor &&) = default;
  Predictor &operator=(Predictor &&) = default;
  Predictor(const Predictor &) = delete;
  Predictor &operator=(const Predictor &) = delete;

  void set_offline(bool offline) { offline_ = offline; }
  bool is_offline() const { return offline_; }

  std::vector<float> step(const std::vector<float> &input,
                          const std::vector<float> *actual = nullptr) {
    std::lock_guard<std::mutex> lock(*mtx_);

    auto h1 = lstm1_.forward(input, /*record_history=*/false);
    auto h2 = lstm2_.forward(h1, /*record_history=*/false);
    
    h2_history_.push_back(h2);
    if ((int)h2_history_.size() > MAX_ATTN_HISTORY) h2_history_.pop_front();

    int T = h2_history_.size();
    auto q = h2;
    std::vector<float> scores(T);
    float scale = 1.0f / std::sqrt((float)hidden_dim);
    float max_s = -1e9f;
    for(int i=0; i<T; i++) {
        float dot = 0.f;
        for(int j=0; j<hidden_dim; j++) dot += q[j] * h2_history_[i][j];
        scores[i] = dot * scale;
        if(scores[i] > max_s) max_s = scores[i];
    }
    float sum_exp = 0.f;
    std::vector<float> alpha(T);
    for(int i=0; i<T; i++) {
        alpha[i] = std::exp(scores[i] - max_s);
        sum_exp += alpha[i];
    }
    std::vector<float> ctx(hidden_dim, 0.f);
    for(int i=0; i<T; i++) {
        alpha[i] /= sum_exp;
        for(int j=0; j<hidden_dim; j++) ctx[j] += alpha[i] * h2_history_[i][j];
    }
    std::vector<float> h_out(hidden_dim);
    for(int j=0; j<hidden_dim; j++) h_out[j] = q[j] + ctx[j];

    auto pred = out_.forward(h_out);

    if (!offline_ && actual != nullptr) {
      int H = hidden_dim, D = out_.output_dim;
      std::vector<float> delta_out(D, 0.f);

      // Find best head
      int best_k = 0;
      float best_dist = 1e9f;
      for(int k=0; k<K; k++) {
          float d = l2(pred.data() + k * (target_dim + 1), actual->data(), target_dim);
          if (d < best_dist) { best_dist = d; best_k = k; }
      }
      last_error_ = best_dist;

      std::vector<int> active_out;
      active_out.reserve(D);
      
      for(int k=0; k<K; k++) {
          int offset = k * (target_dim + 1);
          int conf_idx = offset + target_dim;
          float conf = pred[conf_idx];
          
          if (k == best_k) {
              // Pull towards target
              for(int i=0; i<target_dim; i++) {
                  float diff = pred[offset + i] - (*actual)[i];
                  delta_out[offset + i] = diff; // MSE gradient
                  active_out.push_back(offset + i);
              }
              // Push confidence up
              delta_out[conf_idx] = (conf - 1.f) * conf * (1.f - conf);
              active_out.push_back(conf_idx);
          } else {
              // Push confidence down
              float conf_delta = (conf - 0.f) * conf * (1.f - conf);
              if (std::abs(conf_delta) > 1e-4f) {
                  delta_out[conf_idx] = conf_delta;
                  active_out.push_back(conf_idx);
              }
          }
      }

      std::vector<float> delta_h2_out(H, 0.f);
      for (int j = 0; j < H; j++) {
        float s = 0.f;
        for (int i : active_out)
          s += out_.W[(size_t)i * H + j] * delta_out[i];
        delta_h2_out[j] = s;
      }

      // Backprop through Attention
      std::vector<float> d_q = delta_h2_out;
      std::vector<float> d_ctx = delta_h2_out;

      float sum_alpha_d_alpha = 0.f;
      std::vector<float> d_alpha(T);
      for(int i=0; i<T; i++) {
          float da = 0.f;
          for(int j=0; j<H; j++) da += d_ctx[j] * h2_history_[i][j];
          d_alpha[i] = da;
          sum_alpha_d_alpha += alpha[i] * da;
      }
      
      float d_score_current = alpha[T-1] * (d_alpha[T-1] - sum_alpha_d_alpha);
      for(int j=0; j<H; j++) d_q[j] += d_score_current * h2_history_[T-1][j] * scale;

      std::vector<float> delta_h2 = d_q;
      for(int j=0; j<H; j++) {
          delta_h2[j] += alpha[T-1] * d_ctx[j];
          delta_h2[j] += d_score_current * q[j] * scale;
      }

#ifdef USE_CUDA
      cuda_matvec_sub(out_.W.data(), h_out.data(), delta_out.data(), D, H);
      cuda_device_synchronize();
#else
      for (int i : active_out) {
        float dp = lr_ * delta_out[i];
        float *row = out_.W.data() + (size_t)i * H;
        const float *h_out_p = h_out.data();
        for (int j = 0; j < H; j++)
            row[j] -= dp * h_out_p[j];
        out_.b[i] -= dp;
      }
#endif

      auto delta_h1 = lstm2_.backward(delta_h2, lr_);
      lstm1_.backward(delta_h1, lr_);
    }

    return pred;
  }

  float train_sequence(const std::vector<std::vector<float>> &inputs,
                       const std::vector<float> &target, int n_bptt = -1) {
    std::lock_guard<std::mutex> lock(*mtx_);
    if (inputs.empty()) return 0.f;

    lstm1_.clear_history();
    lstm2_.clear_history();

    std::vector<std::vector<float>> h2_seq;
    for (const auto &inp : inputs) {
      auto h1 = lstm1_.forward(inp, /*record_history=*/true);
      auto h2 = lstm2_.forward(h1, /*record_history=*/true);
      h2_seq.push_back(h2);
    }

    int steps = h2_seq.size();
    auto q = h2_seq.back();
    std::vector<float> scores(steps);
    float scale = 1.0f / std::sqrt((float)hidden_dim);
    float max_s = -1e9f;
    for(int i=0; i<steps; i++) {
        float dot = 0.f;
        for(int j=0; j<hidden_dim; j++) dot += q[j] * h2_seq[i][j];
        scores[i] = dot * scale;
        if(scores[i] > max_s) max_s = scores[i];
    }
    float sum_exp = 0.f;
    std::vector<float> alpha(steps);
    for(int i=0; i<steps; i++) {
        alpha[i] = std::exp(scores[i] - max_s);
        sum_exp += alpha[i];
    }
    std::vector<float> ctx(hidden_dim, 0.f);
    for(int i=0; i<steps; i++) {
        alpha[i] /= sum_exp;
        for(int j=0; j<hidden_dim; j++) ctx[j] += alpha[i] * h2_seq[i][j];
    }
    std::vector<float> h_out(hidden_dim);
    for(int j=0; j<hidden_dim; j++) h_out[j] = q[j] + ctx[j];

    auto pred = out_.forward(h_out);

    int H = hidden_dim, D = out_.output_dim;
    std::vector<float> delta_out(D, 0.f);

    int best_k = 0;
    float best_dist = 1e9f;
    for(int k=0; k<K; k++) {
        float d = l2(pred.data() + k * (target_dim + 1), target.data(), target_dim);
        if (d < best_dist) { best_dist = d; best_k = k; }
    }
    last_error_ = best_dist;

    std::vector<int> active_out;
    active_out.reserve(D);
    
    for(int k=0; k<K; k++) {
        int offset = k * (target_dim + 1);
        int conf_idx = offset + target_dim;
        float conf = pred[conf_idx];
        
        if (k == best_k) {
            for(int i=0; i<target_dim; i++) {
                float diff = pred[offset + i] - target[i];
                delta_out[offset + i] = diff;
                active_out.push_back(offset + i);
            }
            delta_out[conf_idx] = (conf - 1.f) * conf * (1.f - conf);
            active_out.push_back(conf_idx);
        } else {
            float conf_delta = (conf - 0.f) * conf * (1.f - conf);
            if (std::abs(conf_delta) > 1e-4f) {
                delta_out[conf_idx] = conf_delta;
                active_out.push_back(conf_idx);
            }
        }
    }

    std::vector<float> delta_h2_out(H, 0.f);
    for (int j = 0; j < H; j++) {
      float s = 0.f;
      for (int i : active_out)
        s += out_.W[(size_t)i * H + j] * delta_out[i];
      delta_h2_out[j] = s;
    }
    
    // Backprop through attention across full sequence
    std::vector<std::vector<float>> delta_h2_seq(steps, std::vector<float>(H, 0.f));
    std::vector<float> d_q = delta_h2_out;
    std::vector<float> d_ctx = delta_h2_out;

    float sum_alpha_d_alpha = 0.f;
    std::vector<float> d_alpha(steps);
    for(int i=0; i<steps; i++) {
        float da = 0.f;
        for(int j=0; j<H; j++) da += d_ctx[j] * h2_seq[i][j];
        d_alpha[i] = da;
        sum_alpha_d_alpha += alpha[i] * da;
    }

    for(int i=0; i<steps; i++) {
        float d_score_i = alpha[i] * (d_alpha[i] - sum_alpha_d_alpha);
        for(int j=0; j<H; j++) {
            d_q[j] += d_score_i * h2_seq[i][j] * scale;
            delta_h2_seq[i][j] += alpha[i] * d_ctx[j];
            delta_h2_seq[i][j] += d_score_i * q[j] * scale;
        }
    }
    for(int j=0; j<H; j++) delta_h2_seq[steps-1][j] += d_q[j];

#ifdef USE_CUDA
    cuda_matvec_sub(out_.W.data(), h_out.data(), delta_out.data(), D, H);
    cuda_device_synchronize();
#else
#ifdef USE_OPENMP
#endif
    for (int i : active_out) {
      float dp = lr_ * delta_out[i];
      float *row = out_.W.data() + (size_t)i * H;
      const float *h_out_p = h_out.data();
      for (int j = 0; j < H; j++)
        row[j] -= dp * h_out_p[j];
      out_.b[i] -= lr_ * delta_out[i];
    }
#endif

    int unroll_steps = (n_bptt < 0) ? steps : std::min(n_bptt, steps);
    
    std::vector<std::vector<float>> delta_h2_seq_sliced;
    if (unroll_steps < steps) {
        delta_h2_seq_sliced.assign(delta_h2_seq.end() - unroll_steps, delta_h2_seq.end());
    } else {
        delta_h2_seq_sliced = delta_h2_seq;
    }

    auto delta_h1_seq = lstm2_.backward_through_time(delta_h2_seq_sliced, lr_, unroll_steps);
    lstm1_.backward_through_time(delta_h1_seq, lr_, unroll_steps);

    return best_dist;
  }

  void reset() {
    lstm1_.reset_state();
    lstm2_.reset_state();
    h2_history_.clear();
  }

  float last_error() const noexcept { return last_error_; }
  float lr() const noexcept { return lr_; }
  void set_lr(float v) { lr_ = v; }

  void save(const std::string &path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Predictor::save: cannot open " + path);
    f.write((const char *)&input_dim, sizeof(int));
    f.write((const char *)&hidden_dim, sizeof(int));
    f.write((const char *)&target_dim, sizeof(int));
    f.write((const char *)&K, sizeof(int));
    f.write((const char *)&lr_, sizeof(float));
    lstm1_.save(f);
    lstm2_.save(f);
    out_.save(f);
  }

  static Predictor load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Predictor::load: cannot open " + path);
    Predictor p;
    f.read((char *)&p.input_dim, sizeof(int));
    f.read((char *)&p.hidden_dim, sizeof(int));
    f.read((char *)&p.target_dim, sizeof(int));
    f.read((char *)&p.K, sizeof(int));
    f.read((char *)&p.lr_, sizeof(float));
    p.lstm1_ = SparseLSTMLayer::load(f);
    p.lstm2_ = SparseLSTMLayer::load(f);
    p.out_ = OutputLayer::load(f);
    p.mtx_ = std::make_unique<std::mutex>();
    return p;
  }

  void expand_dims(int new_dims) {
      std::lock_guard<std::mutex> lock(*mtx_);
      if (new_dims <= input_dim) return;
      // lstm1_.expand_input_dim(new_dims); // Disabled for Sparse LSTM
      out_.expand_output_dim(new_dims);
      input_dim = new_dims;
      target_dim = new_dims;
  }
};

} // namespace brain2