#pragma once
/*
 * predictor.hpp — LSTM Prediction Engine, Brain v2
 *
 * FIXES APPLIED:
 *   LM Head implementation with Weight Tying against GloVe embeddings.
 *   Softmax cross-entropy over restricted corpus vocabulary.
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
#include <unordered_map>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

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

inline void matvec(const float *__restrict__ W, const float *__restrict__ x,
                   float *__restrict__ out, int m, int n) noexcept {
#ifdef USE_CUDA
    cuda_matvec_add(W, x, out, m, n);
    cuda_device_synchronize();
#elif defined(__APPLE__)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, W, n, x, 1, 1.0f, out, 1);
#else
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

// ── LM Projection Head ─────────────────────────────────────────────────

struct LMHead {
    std::vector<float> W_proj;   // [target_dim * hidden_dim]
    std::vector<float> word_bias; // [vocab_size]
    int hidden_dim_ = 0;
    int target_dim_ = 0;

    // Static embedding table (frozen)
    const float* E_ = nullptr;
    int vocab_size_ = 0;
    const bool* language_frozen_ptr_ = nullptr; // pointer to language.frozen_

    // Active subset caches for BLAS
    std::vector<int> active_vocab_;
    std::unordered_map<int, int> active_idx_; // word_id → index in active_vocab_
    std::vector<float> E_active_; // [active_size * target_dim]
    std::vector<float> bias_active_; // [active_size]

    // ── Adam state for head params (not persisted; lazily re-initialized) ──
    std::vector<float> m_W_, v_W_;   // W_proj moments
    std::vector<float> m_b_, v_b_;   // bias_active_ moments
    int t_W_ = 0, t_b_ = 0;
    // Head learns with Adam at its own rates (LSTM stays on SGD at lr_).
    // bias: its logit effect is direct (+lr per step), and it must traverse
    //   the unigram log-frequency range (~4-5 nats) within one corpus pass.
    // W_proj: gradient of a row is d_h_proj * h_out, so a sign-aligned Adam
    //   step moves the logit by ~lr * sum|h_out_j| (~200 at hidden=512). A
    //   bias-sized lr here explodes CE within one chunk; keep it ~25x smaller.
    float adam_lr_b_ = 0.05f;
    float adam_lr_W_ = 0.002f;

    LMHead() = default;
    LMHead(int hidden_dim, int target_dim, std::mt19937& rng)
        : hidden_dim_(hidden_dim), target_dim_(target_dim) {
        W_proj.resize(target_dim * hidden_dim, 0.f);
        // Near-zero init: logits start at ~0 so CE starts at exactly ln(V)
        // (uniform). Xavier-scale init against unnormalized GloVe embeddings
        // (norm ~5) gives logit std ~3-4, i.e. CE ≈ ln(V) + σ²/2 — several
        // nats WORSE than uniform before training even begins.
        float scale = 0.01f * std::sqrt(2.f / (hidden_dim + target_dim));
        std::normal_distribution<float> d(0.f, scale);
        for(auto& w : W_proj) w = d(rng);
    }

    static void adam_update(float* w, const float* g, float* m, float* v,
                            size_t n, int t, float lr,
                            float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) {
        float bc1 = 1.f - std::pow(b1, (float)t);
        float bc2 = 1.f - std::pow(b2, (float)t);
        for (size_t i = 0; i < n; i++) {
            m[i] = b1 * m[i] + (1.f - b1) * g[i];
            v[i] = b2 * v[i] + (1.f - b2) * g[i] * g[i];
            w[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
        }
    }

    // Adam step on W_proj given dense gradient gW [target_dim_ * hidden_dim_]
    void adam_W(const float* gW) {
        if (m_W_.size() != W_proj.size()) {
            m_W_.assign(W_proj.size(), 0.f);
            v_W_.assign(W_proj.size(), 0.f);
            t_W_ = 0;
        }
        adam_update(W_proj.data(), gW, m_W_.data(), v_W_.data(),
                    W_proj.size(), ++t_W_, adam_lr_W_);
    }

    // Adam step on active bias given gradient gb [active_size]; syncs word_bias
    void adam_bias(const float* gb) {
        size_t n = bias_active_.size();
        if (m_b_.size() != n) {
            m_b_.assign(n, 0.f);
            v_b_.assign(n, 0.f);
            t_b_ = 0;
        }
        adam_update(bias_active_.data(), gb, m_b_.data(), v_b_.data(),
                    n, ++t_b_, adam_lr_b_);
        for (size_t i = 0; i < n; i++)
            word_bias[active_vocab_[i]] = bias_active_[i];
    }

    // O(1) lookup: word_id → active index, or -1 if not in active vocab
    int active_index(int word_id) const {
        auto it = active_idx_.find(word_id);
        return (it == active_idx_.end()) ? -1 : it->second;
    }

    void set_active_vocab(const float* E, int vocab_size, const std::vector<int>& active_vocab, const bool* frozen_ptr) {
        if (target_dim_ <= 0)
            throw std::runtime_error("LMHead::set_active_vocab: target_dim not set (load order bug?)");
        E_ = E;
        vocab_size_ = vocab_size;
        language_frozen_ptr_ = frozen_ptr;
        word_bias.resize(vocab_size, 0.f); // Ensure bias array exists

        active_vocab_ = active_vocab;
        int active_size = active_vocab.size();

        E_active_.resize((size_t)active_size * target_dim_);
        bias_active_.resize(active_size);
        active_idx_.clear();
        active_idx_.reserve(active_size);

        for (int i = 0; i < active_size; i++) {
            int v = active_vocab[i];
            const float* emb = E_ + (size_t)v * target_dim_;
            std::copy(emb, emb + target_dim_, E_active_.begin() + (size_t)i * target_dim_);
            bias_active_[i] = word_bias[v];
            active_idx_[v] = i;
        }
    }
    
    void save(std::ofstream& f) const {
        size_t n = W_proj.size();
        f.write((const char*)&n, sizeof(size_t));
        if (n > 0) f.write((const char*)W_proj.data(), n * sizeof(float));
        
        n = word_bias.size();
        f.write((const char*)&n, sizeof(size_t));
        if (n > 0) f.write((const char*)word_bias.data(), n * sizeof(float));
    }
    
    static LMHead load(std::ifstream& f) {
        LMHead h;
        size_t n;
        f.read((char*)&n, sizeof(size_t));
        if (n > 0) {
            h.W_proj.resize(n);
            f.read((char*)h.W_proj.data(), n * sizeof(float));
        }
        f.read((char*)&n, sizeof(size_t));
        if (n > 0) {
            h.word_bias.resize(n);
            f.read((char*)h.word_bias.data(), n * sizeof(float));
        }
        return h;
    }
};

// ── Predictor ──────────────────────────────────────────────────────────


enum class ErrorMode {
    FULL,
    EMBED_PROXY,
    SKIP
};

class Predictor {
public:
  int input_dim;
  int hidden_dim;
  int target_dim;
  int K;
  float lr_ = 0.01f;
  float last_error_ = 0.f;
  float avg_ce_ = 0.f;
  float avg_err_sq_ = 0.f;
  bool offline_ = false;
  int top_k_report = 20;

  SparseLSTMLayer lstm1_;
  SparseLSTMLayer lstm2_;
  LMHead head_;
  std::unique_ptr<std::mutex> mtx_;
  
  std::deque<std::vector<float>> h2_history_;
  int MAX_ATTN_HISTORY = 50;

  Predictor() : mtx_(std::make_unique<std::mutex>()) {}

  Predictor(int input_dim, int hidden_dim, int target_dim, int K,
            float lr = 0.01f, unsigned int seed = 42)
      : input_dim(input_dim), hidden_dim(hidden_dim), target_dim(target_dim),
        K(K), mtx_(std::make_unique<std::mutex>()) {
    std::mt19937 rng(seed);
    // Use 256 active neurons per layer for sparsity
    lstm1_ = SparseLSTMLayer(input_dim, hidden_dim, 256, rng);
    lstm2_ = SparseLSTMLayer(hidden_dim, hidden_dim, 256, rng);
    head_ = LMHead(hidden_dim, target_dim, rng); // target_dim is embed_dim
  }

  Predictor(Predictor &&) = default;
  Predictor &operator=(Predictor &&) = default;
  Predictor(const Predictor &) = delete;
  Predictor &operator=(const Predictor &) = delete;

  void set_offline(bool offline) { offline_ = offline; }
  bool is_offline() const { return offline_; }
  float perplexity() const { return std::exp(avg_ce_); }
  float surprise_threshold() const { return avg_ce_ + std::sqrt(std::max(0.0f, avg_err_sq_ - avg_ce_ * avg_ce_)); }

  void set_active_vocab(const float* E, int vocab_size, const std::vector<int>& active_vocab, const bool* language_frozen_ptr) {
      std::lock_guard<std::mutex> lock(*mtx_);
      head_.set_active_vocab(E, vocab_size, active_vocab, language_frozen_ptr);
  }

  // Returns vector of pairs <probability, word_id>
  std::vector<std::pair<float, int>> step(const std::vector<float> &input,
                                          int target_word_id = -1,
                                          ErrorMode error_mode = ErrorMode::FULL) {
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

    // Project to Embedding Space
    std::vector<float> h_proj(target_dim, 0.f);
    matvec(head_.W_proj.data(), h_out.data(), h_proj.data(), target_dim, hidden_dim);

    // If context not set, return empty
    if (!head_.E_ || head_.active_vocab_.empty()) return {};

    static int print_count = 0;
    if (print_count < 5) {
        printf("[Predictor::step] Mode: %d, target_word_id: %d\n", (int)error_mode, target_word_id);
        print_count++;
    }

    if (error_mode == ErrorMode::SKIP) {
        static int printed_skip = 0;
        if (printed_skip++ == 0) printf("[Predictor::step] Executing SKIP branch for first pair\n");
        return {};
    }
    
    if (error_mode == ErrorMode::EMBED_PROXY && target_word_id >= 0) {
        static int printed_proxy = 0;
        if (printed_proxy++ == 0) printf("[Predictor::step] Executing EMBED_PROXY branch for first pair\n");
        // Fast cosine proxy
        const float* emb = head_.E_ + target_word_id * target_dim;
        float dot = 0.f, norm_h = 0.f, norm_e = 0.f;
        for (int i = 0; i < target_dim; i++) {
            dot += h_proj[i] * emb[i];
            norm_h += h_proj[i] * h_proj[i];
            norm_e += emb[i] * emb[i];
        }
        float cos_sim = 0.f;
        if (norm_h > 1e-8f && norm_e > 1e-8f) {
            cos_sim = dot / (std::sqrt(norm_h) * std::sqrt(norm_e));
        }
        
        float dist = 1.0f - cos_sim; // [0, 2]
        
        // Removed hand-scaling, using raw distance since we calibrate dynamically now.
        last_error_ = dist;
        
        if (!offline_) {
            avg_ce_ = 0.99f * avg_ce_ + 0.01f * last_error_;
            avg_err_sq_ = 0.99f * avg_err_sq_ + 0.01f * (last_error_ * last_error_);
        }
        return {};
    }

    // Enforce freeze constraint before using cache
    if (head_.language_frozen_ptr_ && !(*head_.language_frozen_ptr_)) {
        throw std::runtime_error("Predictor E_active_ cache is stale! Language embeddings were unfrozen.");
    }

    // Compute Logits over Active Vocabulary
    int active_size = head_.active_vocab_.size();
    std::vector<float> logits = head_.bias_active_; // Pre-fill with bias
    
#ifdef __APPLE__
    // logits = E_active * h_proj + bias_active
    cblas_sgemv(CblasRowMajor, CblasNoTrans, active_size, target_dim, 
                1.0f, head_.E_active_.data(), target_dim, 
                h_proj.data(), 1, 
                1.0f, logits.data(), 1);
    
    float max_logit = -1e9f;
    for (int i = 0; i < active_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
#else
    float max_logit = -1e9f;
    for (int i = 0; i < active_size; i++) {
        int v = head_.active_vocab_[i];
        float dot = 0.f;
        const float* emb = head_.E_ + v * target_dim;
        for (int d = 0; d < target_dim; d++) {
            dot += h_proj[d] * emb[d];
        }
        logits[i] = dot + head_.bias_active_[i];
        if (logits[i] > max_logit) max_logit = logits[i];
    }
#endif

    // Softmax
    std::vector<float> probs(logits.size());
    float sum_prob = 0.f;
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_prob += probs[i];
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_prob;

    // Build top-K return
    std::vector<std::pair<float, int>> ranked_preds;
    for (size_t i = 0; i < probs.size(); i++) {
        ranked_preds.push_back({probs[i], head_.active_vocab_[i]});
    }
    size_t k = std::min((size_t)top_k_report, ranked_preds.size());
    std::partial_sort(ranked_preds.begin(), ranked_preds.begin() + k, ranked_preds.end(), [](auto& a, auto& b){ return a.first > b.first; });
    ranked_preds.resize(k);

    // Compute Loss and Backprop
    if (target_word_id >= 0) {
        int target_idx = head_.active_index(target_word_id);

        if (target_idx >= 0) {
            float p_target = probs[target_idx];
            last_error_ = -std::log(std::max(p_target, 1e-7f)); // Cross-entropy
            avg_ce_ = 0.99f * avg_ce_ + 0.01f * last_error_;

            if (!offline_) {
                // Gradient of CE loss w.r.t logits is (p - y)
                std::vector<float> d_logits = probs;
                d_logits[target_idx] -= 1.0f;

            // Backprop into h_proj: d_h_proj = E_active^T * d_logits
            std::vector<float> d_h_proj(target_dim, 0.f);
            int active_size = head_.active_vocab_.size();

#ifdef __APPLE__
            cblas_sgemv(CblasRowMajor, CblasTrans, active_size, target_dim,
                        1.0f, head_.E_active_.data(), target_dim,
                        d_logits.data(), 1,
                        0.0f, d_h_proj.data(), 1);
#else
            for (int i = 0; i < active_size; i++) {
                float dl = d_logits[i];
                const float* emb = head_.E_active_.data() + (size_t)i * target_dim;
                for (int d = 0; d < target_dim; d++) {
                    d_h_proj[d] += dl * emb[d];
                }
            }
#endif
            head_.adam_bias(d_logits.data());

            // Backprop into W_proj and h_out.
            // delta_h2_out must use W_proj from BEFORE this step's update.
            std::vector<float> delta_h2_out(hidden_dim, 0.f);
            std::vector<float> g_W((size_t)target_dim * hidden_dim, 0.f);
#ifdef __APPLE__
            // delta_h2_out = W_proj^T * d_h_proj
            cblas_sgemv(CblasRowMajor, CblasTrans, target_dim, hidden_dim,
                        1.0f, head_.W_proj.data(), hidden_dim,
                        d_h_proj.data(), 1,
                        0.0f, delta_h2_out.data(), 1);
            // g_W = d_h_proj * h_out^T
            cblas_sger(CblasRowMajor, target_dim, hidden_dim,
                       1.0f, d_h_proj.data(), 1, h_out.data(), 1,
                       g_W.data(), hidden_dim);
#else
            for (int i = 0; i < target_dim; i++) {
                float dp = d_h_proj[i];
                float* gr = g_W.data() + (size_t)i * hidden_dim;
                const float* wr = head_.W_proj.data() + (size_t)i * hidden_dim;
                for (int j = 0; j < hidden_dim; j++) {
                    delta_h2_out[j] += dp * wr[j];
                    gr[j] = dp * h_out[j];
                }
            }
#endif
            head_.adam_W(g_W.data());

            auto delta_h2 = delta_h2_out;
            auto delta_h1 = lstm2_.backward(delta_h2, lr_);
            lstm1_.backward(delta_h1, lr_);
            }
        }
    }

    return ranked_preds;
  }

  float train_lm_sequence(const std::vector<std::vector<float>> &inputs,
                          const std::vector<int> &targets) {
      std::lock_guard<std::mutex> lock(*mtx_);
      if (inputs.empty() || inputs.size() != targets.size() || !head_.E_ || head_.active_vocab_.empty()) return 0.f;

      if (head_.language_frozen_ptr_ && !(*head_.language_frozen_ptr_)) {
          throw std::runtime_error("Predictor E_active_ cache is stale!");
      }

      int active_size = head_.active_vocab_.size();
      float total_ce = 0.f;
      int num_valid = 0;

      int total_T = inputs.size();
      int MAX_T = 128;
      
      for (int chunk_start = 0; chunk_start < total_T; chunk_start += MAX_T) {
          int T = std::min(MAX_T, total_T - chunk_start);
          
          lstm1_.clear_history();
          lstm2_.clear_history();
          
          std::vector<float> H_OUT(T * hidden_dim);
          std::vector<float> H_PROJ(T * target_dim);
          
          for (int t = 0; t < T; t++) {
              int global_t = chunk_start + t;
              auto h1 = lstm1_.forward(inputs[global_t], !offline_);
              auto h2 = lstm2_.forward(h1, !offline_);
              
              h2_history_.push_back(h2);
              if ((int)h2_history_.size() > MAX_ATTN_HISTORY) h2_history_.pop_front();
              
              int H = h2_history_.size();
              auto q = h2;
              std::vector<float> scores(H);
              float scale = 1.0f / std::sqrt((float)hidden_dim);
              float max_s = -1e9f;
              for(int i=0; i<H; i++) {
                  float dot = 0.f;
                  for(int j=0; j<hidden_dim; j++) dot += q[j] * h2_history_[i][j];
                  scores[i] = dot * scale;
                  if(scores[i] > max_s) max_s = scores[i];
              }
              float sum_exp = 0.f;
              std::vector<float> alpha(H);
              for(int i=0; i<H; i++) {
                  alpha[i] = std::exp(scores[i] - max_s);
                  sum_exp += alpha[i];
              }
              std::vector<float> ctx(hidden_dim, 0.f);
              for(int i=0; i<H; i++) {
                  alpha[i] /= sum_exp;
                  for(int j=0; j<hidden_dim; j++) ctx[j] += alpha[i] * h2_history_[i][j];
              }
              std::vector<float> h_out(hidden_dim);
              for(int j=0; j<hidden_dim; j++) h_out[j] = q[j] + ctx[j];
              
              std::copy(h_out.begin(), h_out.end(), H_OUT.begin() + t * hidden_dim);
              
              std::vector<float> h_proj(target_dim, 0.f);
              matvec(head_.W_proj.data(), h_out.data(), h_proj.data(), target_dim, hidden_dim);
              std::copy(h_proj.begin(), h_proj.end(), H_PROJ.begin() + t * target_dim);
          }
          
          std::vector<float> LOGITS(active_size * T);
          for (int i = 0; i < active_size; i++) {
              float b = head_.bias_active_[i];
              for (int t = 0; t < T; t++) {
                  LOGITS[i * T + t] = b;
              }
          }
          
#ifdef __APPLE__
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                      active_size, T, target_dim,
                      1.0f, head_.E_active_.data(), target_dim,
                      H_PROJ.data(), target_dim,
                      1.0f, LOGITS.data(), T);
#else
          for (int i = 0; i < active_size; i++) {
              const float* emb = head_.E_active_.data() + i * target_dim;
              for (int t = 0; t < T; t++) {
                  float dot = 0.f;
                  const float* hp = H_PROJ.data() + t * target_dim;
                  for (int d = 0; d < target_dim; d++) {
                      dot += emb[d] * hp[d];
                  }
                  LOGITS[i * T + t] += dot;
              }
          }
#endif

          std::vector<float> D_LOGITS(active_size * T, 0.f);
          for (int t = 0; t < T; t++) {
              int target_word_id = targets[chunk_start + t];
              if (target_word_id < 0) continue;

              int target_idx = head_.active_index(target_word_id);
              if (target_idx < 0) continue;
              
              float max_logit = -1e9f;
              for (int i = 0; i < active_size; i++) {
                  float l = LOGITS[i * T + t];
                  if (l > max_logit) max_logit = l;
              }
              
              float sum_prob = 0.f;
              for (int i = 0; i < active_size; i++) {
                  float p = std::exp(LOGITS[i * T + t] - max_logit);
                  D_LOGITS[i * T + t] = p;
                  sum_prob += p;
              }
              
              float p_target = D_LOGITS[target_idx * T + t] / sum_prob;
              float err = -std::log(std::max(p_target, 1e-7f));
              total_ce += err;
              num_valid++;
              last_error_ = err;
              avg_ce_ = 0.99f * avg_ce_ + 0.01f * last_error_;
              avg_err_sq_ = 0.99f * avg_err_sq_ + 0.01f * (last_error_ * last_error_);
              
              if (!offline_) {
                  for (int i = 0; i < active_size; i++) {
                      D_LOGITS[i * T + t] = (D_LOGITS[i * T + t] / sum_prob);
                  }
                  D_LOGITS[target_idx * T + t] -= 1.0f;
              }
          }
          
          if (offline_) continue;
          
          std::vector<float> D_H_PROJ(T * target_dim, 0.f);
          
#ifdef __APPLE__
          cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      T, target_dim, active_size,
                      1.0f, D_LOGITS.data(), T,
                      head_.E_active_.data(), target_dim,
                      0.0f, D_H_PROJ.data(), target_dim);
          
          std::vector<float> g_bias(active_size, 0.f);
          for (int i = 0; i < active_size; i++) {
              for (int t = 0; t < T; t++) g_bias[i] += D_LOGITS[i * T + t];
          }
          head_.adam_bias(g_bias.data());

          // D_H_OUT must use W_proj from BEFORE this chunk's update
          std::vector<float> D_H_OUT(T * hidden_dim, 0.f);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      T, hidden_dim, target_dim,
                      1.0f, D_H_PROJ.data(), target_dim,
                      head_.W_proj.data(), hidden_dim,
                      0.0f, D_H_OUT.data(), hidden_dim);

          std::vector<float> g_W((size_t)target_dim * hidden_dim, 0.f);
          cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      target_dim, hidden_dim, T,
                      1.0f, D_H_PROJ.data(), target_dim,
                      H_OUT.data(), hidden_dim,
                      0.0f, g_W.data(), hidden_dim);
          head_.adam_W(g_W.data());
#else
          for (int t = 0; t < T; t++) {
              float* dhp = D_H_PROJ.data() + t * target_dim;
              for (int i = 0; i < active_size; i++) {
                  float dl = D_LOGITS[i * T + t];
                  const float* emb = head_.E_active_.data() + i * target_dim;
                  for (int d = 0; d < target_dim; d++) dhp[d] += dl * emb[d];
              }
          }
          std::vector<float> g_bias(active_size, 0.f);
          for (int i = 0; i < active_size; i++) {
              for (int t = 0; t < T; t++) g_bias[i] += D_LOGITS[i * T + t];
          }
          head_.adam_bias(g_bias.data());

          // D_H_OUT must use W_proj from BEFORE this chunk's update
          std::vector<float> D_H_OUT(T * hidden_dim, 0.f);
          for (int t = 0; t < T; t++) {
              float* dho = D_H_OUT.data() + t * hidden_dim;
              float* dhp = D_H_PROJ.data() + t * target_dim;
              for (int d = 0; d < target_dim; d++) {
                  float dp = dhp[d];
                  const float* w_row = head_.W_proj.data() + d * hidden_dim;
                  for (int j = 0; j < hidden_dim; j++) dho[j] += dp * w_row[j];
              }
          }
          std::vector<float> g_W((size_t)target_dim * hidden_dim, 0.f);
          for (int d = 0; d < target_dim; d++) {
              float* gr = g_W.data() + (size_t)d * hidden_dim;
              for (int t = 0; t < T; t++) {
                  float dp = D_H_PROJ[t * target_dim + d];
                  const float* ho = H_OUT.data() + t * hidden_dim;
                  for (int j = 0; j < hidden_dim; j++) gr[j] += dp * ho[j];
              }
          }
          head_.adam_W(g_W.data());
#endif

          std::vector<std::vector<float>> d_h2_seq(T, std::vector<float>(hidden_dim));
          for (int t = 0; t < T; t++) {
              std::copy(D_H_OUT.begin() + t * hidden_dim, 
                        D_H_OUT.begin() + (t + 1) * hidden_dim, 
                        d_h2_seq[t].begin());
          }
          auto delta_h1_seq = lstm2_.backward_through_time(d_h2_seq, lr_, T);
          lstm1_.backward_through_time(delta_h1_seq, lr_, T);
      }
      
      return (num_valid > 0) ? (total_ce / num_valid) : 0.f;
  }

  // Like train_lm_sequence but also returns per-token CE for the fused cognitive pass.
  // ce_per_token[t] = CE at position t, or -1.f if no valid target at t.
  float train_lm_sequence_per_token(const std::vector<std::vector<float>>& inputs,
                                     const std::vector<int>& targets,
                                     std::vector<float>& ce_per_token) {
      ce_per_token.assign(inputs.size(), -1.f);
      std::lock_guard<std::mutex> lock(*mtx_);
      if (inputs.empty() || inputs.size() != targets.size() || !head_.E_ || head_.active_vocab_.empty()) return 0.f;
      if (head_.language_frozen_ptr_ && !(*head_.language_frozen_ptr_))
          throw std::runtime_error("Predictor E_active_ cache is stale!");

      int active_size = (int)head_.active_vocab_.size();
      float total_ce = 0.f;
      int num_valid = 0;
      int total_T = (int)inputs.size();
      int MAX_T = 128;

      for (int chunk_start = 0; chunk_start < total_T; chunk_start += MAX_T) {
          int T = std::min(MAX_T, total_T - chunk_start);
          lstm1_.clear_history();
          lstm2_.clear_history();

          std::vector<float> H_OUT(T * hidden_dim);
          std::vector<float> H_PROJ(T * target_dim);

          for (int t = 0; t < T; t++) {
              int gt = chunk_start + t;
              auto h1 = lstm1_.forward(inputs[gt], !offline_);
              auto h2 = lstm2_.forward(h1, !offline_);
              h2_history_.push_back(h2);
              if ((int)h2_history_.size() > MAX_ATTN_HISTORY) h2_history_.pop_front();
              int H = (int)h2_history_.size();
              auto q = h2;
              float scale = 1.0f / std::sqrt((float)hidden_dim);
              float max_s = -1e9f;
              std::vector<float> scores(H);
              for (int i = 0; i < H; i++) {
                  float dot = 0.f;
                  for (int j = 0; j < hidden_dim; j++) dot += q[j] * h2_history_[i][j];
                  scores[i] = dot * scale;
                  if (scores[i] > max_s) max_s = scores[i];
              }
              float sum_exp = 0.f;
              std::vector<float> alpha(H);
              for (int i = 0; i < H; i++) { alpha[i] = std::exp(scores[i] - max_s); sum_exp += alpha[i]; }
              std::vector<float> ctx(hidden_dim, 0.f);
              for (int i = 0; i < H; i++) {
                  alpha[i] /= sum_exp;
                  for (int j = 0; j < hidden_dim; j++) ctx[j] += alpha[i] * h2_history_[i][j];
              }
              std::vector<float> h_out(hidden_dim);
              for (int j = 0; j < hidden_dim; j++) h_out[j] = q[j] + ctx[j];
              std::copy(h_out.begin(), h_out.end(), H_OUT.begin() + t * hidden_dim);
              std::vector<float> h_proj(target_dim, 0.f);
              matvec(head_.W_proj.data(), h_out.data(), h_proj.data(), target_dim, hidden_dim);
              std::copy(h_proj.begin(), h_proj.end(), H_PROJ.begin() + t * target_dim);
          }

          std::vector<float> LOGITS(active_size * T);
          for (int i = 0; i < active_size; i++) {
              float b = head_.bias_active_[i];
              for (int t = 0; t < T; t++) LOGITS[i * T + t] = b;
          }
#ifdef __APPLE__
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                      active_size, T, target_dim,
                      1.0f, head_.E_active_.data(), target_dim,
                      H_PROJ.data(), target_dim,
                      1.0f, LOGITS.data(), T);
#else
          for (int i = 0; i < active_size; i++) {
              const float* emb = head_.E_active_.data() + i * target_dim;
              for (int t = 0; t < T; t++) {
                  float dot = 0.f;
                  const float* hp = H_PROJ.data() + t * target_dim;
                  for (int d = 0; d < target_dim; d++) dot += emb[d] * hp[d];
                  LOGITS[i * T + t] += dot;
              }
          }
#endif
          std::vector<float> D_LOGITS(active_size * T, 0.f);
          for (int t = 0; t < T; t++) {
              int gt = chunk_start + t;
              int twid = targets[gt];
              if (twid < 0) continue;
              int tidx = head_.active_index(twid);
              if (tidx < 0) continue;
              float max_logit = -1e9f;
              for (int i = 0; i < active_size; i++) { float l = LOGITS[i*T+t]; if (l > max_logit) max_logit = l; }
              float sum_prob = 0.f;
              for (int i = 0; i < active_size; i++) { float p = std::exp(LOGITS[i*T+t]-max_logit); D_LOGITS[i*T+t]=p; sum_prob+=p; }
              float p_target = D_LOGITS[tidx*T+t] / sum_prob;
              float err = -std::log(std::max(p_target, 1e-7f));
              total_ce += err; num_valid++;
              ce_per_token[gt] = err;
              last_error_ = err;
              avg_ce_ = 0.99f * avg_ce_ + 0.01f * err;
              avg_err_sq_ = 0.99f * avg_err_sq_ + 0.01f * err * err;
              if (!offline_) {
                  for (int i = 0; i < active_size; i++) D_LOGITS[i*T+t] /= sum_prob;
                  D_LOGITS[tidx*T+t] -= 1.0f;
              }
          }
          if (offline_) continue;

          std::vector<float> D_H_PROJ(T * target_dim, 0.f);
#ifdef __APPLE__
          cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      T, target_dim, active_size,
                      1.0f, D_LOGITS.data(), T, head_.E_active_.data(), target_dim,
                      0.0f, D_H_PROJ.data(), target_dim);
          std::vector<float> g_bias(active_size, 0.f);
          for (int i = 0; i < active_size; i++) {
              for (int t = 0; t < T; t++) g_bias[i] += D_LOGITS[i*T+t];
          }
          head_.adam_bias(g_bias.data());

          // D_H_OUT must use W_proj from BEFORE this chunk's update
          std::vector<float> D_H_OUT(T * hidden_dim, 0.f);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      T, hidden_dim, target_dim,
                      1.0f, D_H_PROJ.data(), target_dim, head_.W_proj.data(), hidden_dim,
                      0.0f, D_H_OUT.data(), hidden_dim);

          std::vector<float> g_W((size_t)target_dim * hidden_dim, 0.f);
          cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      target_dim, hidden_dim, T,
                      1.0f, D_H_PROJ.data(), target_dim, H_OUT.data(), hidden_dim,
                      0.0f, g_W.data(), hidden_dim);
          head_.adam_W(g_W.data());
#else
          for (int t = 0; t < T; t++) {
              float* dhp = D_H_PROJ.data() + t * target_dim;
              for (int i = 0; i < active_size; i++) {
                  float dl = D_LOGITS[i*T+t];
                  const float* emb = head_.E_active_.data() + i * target_dim;
                  for (int d = 0; d < target_dim; d++) dhp[d] += dl * emb[d];
              }
          }
          std::vector<float> g_bias(active_size, 0.f);
          for (int i = 0; i < active_size; i++) {
              for (int t = 0; t < T; t++) g_bias[i] += D_LOGITS[i*T+t];
          }
          head_.adam_bias(g_bias.data());

          // D_H_OUT must use W_proj from BEFORE this chunk's update
          std::vector<float> D_H_OUT(T * hidden_dim, 0.f);
          for (int t = 0; t < T; t++) {
              float* dho = D_H_OUT.data() + t * hidden_dim;
              float* dhp = D_H_PROJ.data() + t * target_dim;
              for (int d = 0; d < target_dim; d++) {
                  const float* wr = head_.W_proj.data() + d * hidden_dim;
                  float dp = dhp[d];
                  for (int j = 0; j < hidden_dim; j++) dho[j] += dp * wr[j];
              }
          }
          std::vector<float> g_W((size_t)target_dim * hidden_dim, 0.f);
          for (int d = 0; d < target_dim; d++) {
              float* gr = g_W.data() + (size_t)d * hidden_dim;
              for (int t = 0; t < T; t++) {
                  float dp = D_H_PROJ[t*target_dim+d];
                  const float* ho = H_OUT.data() + t * hidden_dim;
                  for (int j = 0; j < hidden_dim; j++) gr[j] += dp * ho[j];
              }
          }
          head_.adam_W(g_W.data());
#endif
          std::vector<std::vector<float>> d_h2_seq(T, std::vector<float>(hidden_dim));
          for (int t = 0; t < T; t++)
              std::copy(D_H_OUT.begin()+t*hidden_dim, D_H_OUT.begin()+(t+1)*hidden_dim, d_h2_seq[t].begin());
          auto dh1 = lstm2_.backward_through_time(d_h2_seq, lr_, T);
          lstm1_.backward_through_time(dh1, lr_, T);
      }
      return (num_valid > 0) ? (total_ce / num_valid) : 0.f;
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
    head_.save(f);
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
    p.head_ = LMHead::load(f);
    p.head_.hidden_dim_ = p.hidden_dim;
    p.head_.target_dim_ = p.target_dim;
    p.mtx_ = std::make_unique<std::mutex>();
    return p;
  }

  
  std::vector<float> get_embedding(int wid) const {
      if (!head_.E_ || wid < 0 || wid >= head_.vocab_size_) return std::vector<float>(target_dim, 0.f);
      const float* e = head_.E_ + wid * target_dim;
      return std::vector<float>(e, e + target_dim);
  }

  void expand_dims(int new_dims) {
      std::lock_guard<std::mutex> lock(*mtx_);
      if (new_dims <= input_dim) return;
      // Resizing the LSTM weight matrices and LM head in place is not implemented.
      // Silently reassigning dims (the old behavior) corrupted the model: every
      // matvec would read W with the wrong stride. Fail loudly instead.
      throw std::runtime_error(
          "Predictor::expand_dims: not implemented — construct a new Predictor "
          "with the desired dims and retrain (or implement weight resizing).");
  }
};

} // namespace brain2
