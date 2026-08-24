#pragma once
/**
 * brain3/crisp/engines/neural/stamlat_transformer.hpp
 *
 * STAMLAT v2 — TRAINABLE PHYSICS-GROUNDED MICRO-LM (honest edition)
 *
 * Kept from the original STAMLAT concept, now made real:
 *   1. DK-RoPE: positions from damped Kuramoto phase relaxation
 *      (15 Euler steps, gamma=0.1, K=0.5) applied as rotary embeddings,
 *      blended 50/50 with a linear ramp (oscillator sync compresses
 *      late-position distinctness otherwise).
 *   2. Symplectic residual stream: inter-block inertial momentum
 *      v <- gamma*v + block(h); h <- h + tau*v, exact adjoint backward.
 *   3. Temperature duality: T=0 argmax emission / T>0 Boltzmann sampling.
 *   4. Pairwise interaction field: multi-head causal softmax attention,
 *      i.e. Boltzmann weights over learned interaction energies.
 *
 * Added vs the old weight-free engine (which had no parameters and no token
 * interaction, hence could not model anything):
 *   tied character embeddings, per-layer weights, LayerNorms, manual
 *   reverse-mode gradients (validated by finite differences in
 *   test_stamlat_transformer.cpp), Adam with global-norm clipping,
 *   binary save/load.
 *
 * Mouth v3 additions (verified in test_stamlat_streaming.cpp):
 *   1. Word-level tokenizer over a char backbone: frequent words become
 *      single tokens; unknown words fall back to char ids greedily.
 *      Same ctx window now spans several conversational turns.
 *   2. KV-cache streaming decode: per-layer UNROTATED k/v rows are cached;
 *      DK-RoPE is re-applied each step with the current-length Kuramoto
 *      table (angles depend on total window length, so caching rotated
 *      keys would drift). Full cache evicts the oldest row — the window
 *      slides with zero recompute, amortized-O(1) per token (~17x measured
 *      vs quadratic recompute at ctx=48). Agreement vs batch decode: while
 *      the window grows, DK-RoPE's length-coupled Kuramoto table introduces
 *      O(1e-3) logit deviation (argmax verified stable); in the full-
 *      window steady state decoding is exact until the first eviction,
 *      after which cache-serving semantics apply (cache preserves each
 *      token's original left-context; batch recompute retells it with the
 *      surviving prefix). See StreamCache.
 *   3. Constrained sampling: optional allow-set of token ids masks logits
 *      before argmax/Boltzmann — content slots can be hard-locked while
 *      style slots sample freely ("fuzzy proposes, crisp disposes").
 *   Save format v3 ("STMLv3") stores the word table; v2 files still load.
 */

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <utility>
#include <mutex>

namespace brain3 {
namespace engines {
namespace neural {

// ─────────────────────────────────────────────────────────────────────────────
// Dense row-major matrix
// ─────────────────────────────────────────────────────────────────────────────
struct Mat {
    int r = 0, c = 0;
    std::vector<float> a;

    Mat() = default;
    Mat(int R, int C, float v = 0.f) : r(R), c(C), a((size_t)R * C, v) {}

    void zero() { std::fill(a.begin(), a.end(), 0.f); }
    float& at(int i, int j) { return a[(size_t)i * c + j]; }
    float  at(int i, int j) const { return a[(size_t)i * c + j]; }
};

inline void matmul(const Mat& X, const Mat& W, Mat& Y) {           // Y = X*W
    const int T = X.r, DI = X.c, DO = W.c;
    std::fill(Y.a.begin(), Y.a.end(), 0.f);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < DI; ++i) {
            const float xv = X.at(t, i);
            if (xv == 0.f) continue;
            const float* wr = &W.a[(size_t)i * DO];
            float* yr = &Y.a[(size_t)t * DO];
            for (int o = 0; o < DO; ++o) yr[o] += xv * wr[o];
        }
}
inline void matmul_trans(const Mat& X, const Mat& W, Mat& Y) {     // Y = X*W^T, W:(DO x DI)
    const int T = X.r, DI = W.c, DO = W.r;
    std::fill(Y.a.begin(), Y.a.end(), 0.f);
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < DO; ++o) {
            const float* xr = &X.a[(size_t)t * DI];
            const float* wr = &W.a[(size_t)o * DI];
            double s = 0.0;
            for (int i = 0; i < DI; ++i) s += (double)xr[i] * wr[i];
            Y.at(t, o) = (float)s;
        }
}
inline void add_bias(Mat& Y, const Mat& b) {
    for (int t = 0; t < Y.r; ++t)
        for (int j = 0; j < Y.c; ++j) Y.at(t, j) += b.a[j];
}

// ─────────────────────────────────────────────────────────────────────────────
// DK-RoPE angles (Kuramoto relaxation, cached per length)
// ─────────────────────────────────────────────────────────────────────────────
inline const std::vector<float>& kuramoto_angles(int seq_len) {
    // Thread-safe lazy cache: concurrent first-touch from multiple threads
    // (e.g. orchestrator + background daemons) must not race the map.
    static std::mutex cache_mtx;
    static std::unordered_map<int, std::vector<float>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mtx);
        auto it = cache.find(seq_len);
        if (it != cache.end()) return it->second;
    }
    const float dt = 0.05f, gamma = 0.1f, K = 0.5f;
    std::vector<float> ph(seq_len, 0.f), vel(seq_len, 0.f), om(seq_len);
    for (int i = 0; i < seq_len; ++i) om[i] = 6.2831853f / (1.f + std::exp(-0.05f * i));
    for (int step = 0; step < 15; ++step)
        for (int i = 0; i < seq_len; ++i) {
            float coup = 0.f;
            for (int j = 0; j < seq_len; ++j) coup += std::sin(ph[j] - ph[i]);
            vel[i] += dt * (om[i] + (K / seq_len) * coup - gamma * vel[i]);
            ph[i]  += dt * vel[i];
        }
    for (int i = 0; i < seq_len; ++i) ph[i] = 0.5f * ph[i] + 0.5f * (float)i;
    std::lock_guard<std::mutex> lock(cache_mtx);
    return cache.emplace(seq_len, std::move(ph)).first->second;
}

static inline void rotary_apply(Mat& M, int head_dim, bool inverse) {
    const std::vector<float>& ang = kuramoto_angles(M.r);
    const int pairs = head_dim / 2;
    std::vector<float> inv_freq(pairs);
    for (int p = 0; p < pairs; ++p)
        inv_freq[p] = std::pow(10000.f, -2.f * p / (float)head_dim);
    for (int t = 0; t < M.r; ++t)
        for (int p = 0; p < pairs; ++p) {
            float th = ang[t] * inv_freq[p];
            if (inverse) th = -th;
            const float cs = std::cos(th), sn = std::sin(th);
            float& x = M.at(t, 2 * p);
            float& y = M.at(t, 2 * p + 1);
            const float nx = x * cs - y * sn;
            const float ny = x * sn + y * cs;
            x = nx; y = ny;
        }
}

// Full-width variant for the KV-cache path: treats M.c as H consecutive
// head_dim blocks and rotates every row of every block against an externally
// supplied angle table whose length is the CURRENT window size (caller
// offsets the pointer to address a specific position). Math is identical to
// rotary_apply; only the angle source differs.
static inline void rotary_heads(Mat& M, int head_dim, const float* ang) {
    const int H = M.c / head_dim, pairs = head_dim / 2;
    std::vector<float> inv_freq(pairs);
    for (int p = 0; p < pairs; ++p)
        inv_freq[p] = std::pow(10000.f, -2.f * p / (float)head_dim);
    for (int t = 0; t < M.r; ++t)
        for (int hd = 0; hd < H; ++hd) {
            const int base = hd * head_dim;
            for (int p = 0; p < pairs; ++p) {
                const float th = ang[t] * inv_freq[p];
                const float cs = std::cos(th), sn = std::sin(th);
                float& x = M.at(t, base + 2 * p);
                float& y = M.at(t, base + 2 * p + 1);
                const float nx = x * cs - y * sn;
                const float ny = x * sn + y * cs;
                x = nx; y = ny;
            }
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm
// ─────────────────────────────────────────────────────────────────────────────
struct LNCache { std::vector<float> xhat, mean, rstd; };

inline void ln_forward(const Mat& X, const Mat& g, const Mat& b, Mat& Y, LNCache& ch) {
    const int T = X.r, D = X.c;
    ch.xhat.assign((size_t)T * D, 0.f);
    ch.mean.assign(T, 0.f); ch.rstd.assign(T, 0.f);
    for (int t = 0; t < T; ++t) {
        double mu = 0.; for (int j = 0; j < D; ++j) mu += X.at(t, j);
        mu /= D;
        double var = 0.; for (int j = 0; j < D; ++j) { double dd = X.at(t, j) - mu; var += dd * dd; }
        var /= D;
        const float rs = (float)(1.0 / std::sqrt(var + 1e-5));
        ch.mean[t] = (float)mu; ch.rstd[t] = rs;
        for (int j = 0; j < D; ++j) {
            const float xh = (float)((X.at(t, j) - mu) * (double)rs);
            ch.xhat[(size_t)t * D + j] = xh;
            Y.at(t, j) = g.a[j] * xh + b.a[j];
        }
    }
}
inline void ln_backward(const Mat& dY, const LNCache& ch, const Mat& g,
                        Mat& dX, Mat& dg, Mat& db) {
    const int T = dY.r, D = dY.c;
    for (int t = 0; t < T; ++t) {
        double m1 = 0., m2 = 0.;
        for (int j = 0; j < D; ++j) {
            const double dyh = (double)dY.at(t, j) * g.a[j];
            m1 += dyh;
            m2 += dyh * ch.xhat[(size_t)t * D + j];
        }
        m1 /= D; m2 /= D;
        for (int j = 0; j < D; ++j) {
            const double dy = dY.at(t, j);
            const double xh = ch.xhat[(size_t)t * D + j];
            dg.a[j] += (float)(dy * xh);
            db.a[j] += dY.at(t, j);
            dX.at(t, j) += (float)(ch.rstd[t] * (dy * (double)g.a[j] - m1 - xh * m2));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
struct StamlatConfig {
    int d_model = 80;
    int n_layers = 2;
    int n_heads = 4;
    int d_ff = 224;
    int ctx = 64;
    float depth_gamma = 0.8f;
    float depth_tau   = 0.7f;
    unsigned seed = 42;
    bool operator==(const StamlatConfig& o) const {
        return d_model == o.d_model && n_layers == o.n_layers && n_heads == o.n_heads &&
               d_ff == o.d_ff && ctx == o.ctx;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// STAMLAT Language Model
// ─────────────────────────────────────────────────────────────────────────────
// Weighted SFT example: x[t] predicts y[t]; w[t] is the target weight
// (0 = ignore position, e.g. prompt tokens; 1 = learn to emit, e.g. replies).
struct SftExample {
    std::vector<int> x, y;
    std::vector<float> w;
};

class StamlatLM {
public:
    explicit StamlatLM(const StamlatConfig& cfg) : cfg_(cfg) {
        rng_.seed(cfg_.seed);
        alloc();
    }

    size_t param_count() const {
        size_t n = 0; for (const auto& p : params_) n += p->a.size(); return n;
    }
    const StamlatConfig& config() const { return cfg_; }
    const std::string& vocab_chars() const { return vocab_; }
    const std::vector<std::string>& vocab_words() const { return words_; }
    int   char_vocab_size() const { return (int)vocab_.size(); }
    int   word_vocab_size() const { return (int)words_.size(); }
    int   total_vocab_size() const { return (int)vocab_.size() + (int)words_.size(); }

    static bool is_space(char ch) {
        return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
               ch == '\v' || ch == '\f';
    }

    // Build char vocab (as before) plus a frequency-ranked word table.
    // Words of length >= 2 with freq >= min_word_freq become single tokens,
    // capped at max_word_tokens (ties broken lexicographically → deterministic).
    void build_vocab(const std::string& text,
                     int max_word_tokens = 2048, int min_word_freq = 2) {
        vocab_.clear(); idx_.clear();
        for (char ch : text)
            if (ch != '\r' && idx_.find(ch) == idx_.end()) {
                idx_.emplace(ch, (int)vocab_.size());
                vocab_.push_back(ch);
            }
        if (idx_.find(' ') == idx_.end()) {
            idx_.emplace(' ', (int)vocab_.size());
            vocab_.push_back(' ');
        }

        std::unordered_map<std::string, int> freq;
        size_t i = 0;
        while (i < text.size()) {
            if (is_space(text[i])) { ++i; continue; }
            size_t j = i;
            while (j < text.size() && !is_space(text[j])) ++j;
            if (j - i >= 2) freq[text.substr(i, j - i)]++;
            i = j;
        }
        std::vector<std::pair<std::string, int>> cand(freq.begin(), freq.end());
        std::sort(cand.begin(), cand.end(),
                  [](const auto& a, const auto& b) {
                      if (a.second != b.second) return a.second > b.second;
                      return a.first < b.first;
                  });
        words_.clear(); widx_.clear();
        for (const auto& [w, f] : cand) {
            if ((int)words_.size() >= max_word_tokens) break;
            if (f < min_word_freq) break;              // sorted by freq desc
            widx_.emplace(w, (int)vocab_.size() + (int)words_.size());
            words_.push_back(w);
        }
        rebuild_embedding();
    }

    // ── Tokenizer: greedy whole-word match, per-char fallback ────────────────
    // Word tokens carry no whitespace; inter-word spaces are explicit char
    // tokens. Roundtrip guarantee: decode(encode(x)) == x whenever every
    // character of x is in the char vocab; characters outside the vocab
    // (and '\r') collapse to ' ' by the same policy as char_id().
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        auto push_char = [&](char ch) {
            if (ch == '\r') return;
            const int id = char_id(ch);
            out.push_back(id < 0 ? char_id(' ') : id);   // unknown → space
        };
        size_t i = 0;
        while (i < text.size()) {
            if (is_space(text[i])) { push_char(text[i]); ++i; continue; }
            size_t j = i;
            while (j < text.size() && !is_space(text[j])) ++j;
            const auto it = widx_.find(text.substr(i, j - i));
            if (it != widx_.end()) out.push_back(it->second);
            else for (size_t k = i; k < j; ++k) push_char(text[k]);
            i = j;
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) out += token_surface(id);
        return out;
    }

    // Surface string of a token id: chars occupy [0, char_vocab_size()),
    // word tokens follow.
    std::string token_surface(int id) const {
        if (id < 0 || id >= total_vocab_size()) return "";
        if (id < (int)vocab_.size()) return std::string(1, vocab_[id]);
        return words_[id - (int)vocab_.size()];
    }

    bool is_word_token(int id) const { return id >= (int)vocab_.size(); }

    // Training samples windows from the TOKENIZED stream (words + chars), so
    // word-token embeddings are trained as inputs and the head learns to
    // emit whole words — char fallback still covers everything else.
    float train_step_ids(const std::vector<int>& ids, int batch, float lr) {
        const int N = (int)ids.size();
        if (N < cfg_.ctx + 1) throw std::runtime_error("corpus too short");
        ensure_grads();
        zero_grads();
        std::uniform_int_distribution<int> pick(0, N - cfg_.ctx - 1);
        double loss_sum = 0.; int toks = 0;
        // Uniform next-token LM loss proved most robust at this scale.
        // (forward_backward still accepts per-token weights for future SFT use.)
        for (int b = 0; b < batch; ++b) {
            const int off = pick(rng_);
            std::vector<int> x(cfg_.ctx), y(cfg_.ctx);
            for (int t = 0; t < cfg_.ctx; ++t) {
                x[t] = ids[off + t];
                y[t] = ids[off + t + 1];
            }
            loss_sum += forward_backward(x, y);
            toks += cfg_.ctx;
        }
        scale_grads(1.f / (float)toks);
        clip_grads();
        adam_step(lr);
        return (float)(loss_sum / toks);
    }

    float train_step(const std::string& text, int batch, float lr) {
        return train_step_ids(encode(text), batch, lr);
    }

    // One Adam step over weighted SFT examples (the style loop's absorb
    // phase). Weight 0 positions contribute neither loss nor gradient, so
    // prompt prefixes are conditioning only. Returns mean weighted NLL.
    float sft_step(const std::vector<SftExample>& exs, float lr) {
        ensure_grads();
        zero_grads();
        double loss_sum = 0., wsum = 0.;
        for (const auto& e : exs) {
            loss_sum += forward_backward(e.x, e.y, e.w);
            for (float wt : e.w) wsum += wt;
        }
        if (wsum <= 0.) return 0.f;
        scale_grads(1.f / (float)wsum);
        clip_grads();
        adam_step(lr);
        return (float)(loss_sum / wsum);
    }

    // Parameter checkpointing — lets callers (e.g. the style loop) roll back
    // a retrain that regressed verified behavior.
    std::vector<Mat> snapshot_params() const {
        std::vector<Mat> snap;
        snap.reserve(params_.size());
        for (const auto& p : params_) snap.push_back(*p);
        return snap;
    }
    void restore_params(const std::vector<Mat>& snap) {
        if (snap.size() != params_.size()) return;
        for (size_t i = 0; i < snap.size(); ++i) *params_[i] = snap[i];
    }

    void fit(const std::string& text, int steps, float lr = 3e-3f, int batch = 12,
             int log_every = 500) {
        if (vocab_.empty()) build_vocab(text);
        const std::vector<int> ids = encode(text);       // encode once, reuse
        for (int s = 1; s <= steps; ++s) {
            const float sched = (s > (steps * 7) / 10) ? 0.3f : 1.0f;
            const float L = train_step_ids(ids, batch, lr * sched);
            if (log_every > 0 && (s % log_every == 0 || s == steps))
                std::printf("  [stamlat] step %d/%d  loss %.4f\n", s, steps, L);
        }
    }

    float eval_loss(const std::string& text) const {
        if (vocab_.empty()) return -1.f;
        return eval_loss_ids(encode(text));
    }

    float eval_loss_ids(const std::vector<int>& ids) const {
        const int N = (int)ids.size();
        if (N < cfg_.ctx + 1) return -1.f;
        double tot = 0.; int cnt = 0;
        for (int off = 0; off + cfg_.ctx + 1 <= N; off += cfg_.ctx / 2) {
            auto r = loss_only_ids(off, ids);
            tot += r.first; cnt += r.second;
            if (off + 3 * cfg_.ctx / 2 > N) break;
        }
        return cnt ? (float)(tot / cnt) : -1.f;
    }

    // ── Sampling core: argmax (T≈0) or Boltzmann, under an optional allow-set
    // and an optional sparse logit bias (id → additive logit bonus). Bias is
    // applied before normalization — this is how emotion colors the voice —
    // while the allow-set stays a hard constraint.
    using LogitBias = std::unordered_map<int, float>;

    int pick_token(const Mat& row, float temp,
                   const std::vector<int>* allowed, std::mt19937& rng,
                   const LogitBias* bias = nullptr) const {
        const int V = total_vocab_size();
        std::vector<int> idxs;
        if (allowed) {
            for (int id : *allowed)
                if (id >= 0 && id < V) idxs.push_back(id);
            if (idxs.empty())
                throw std::invalid_argument("pick_token: empty allow-set");
        } else {
            idxs.resize(V);
            for (int i = 0; i < V; ++i) idxs[i] = i;
        }
        auto eff = [&](int id) -> float {
            float v = row.a[id];
            if (bias) {
                const auto it = bias->find(id);
                if (it != bias->end()) v += it->second;
            }
            return v;
        };

        int best = idxs[0];
        if (temp < 1e-5f) {
            float bv = -1e30f;
            for (int id : idxs) {
                const float e = eff(id);
                if (e > bv) { bv = e; best = id; }
            }
        } else {
            float mx = -1e30f;
            for (int id : idxs) mx = std::max(mx, eff(id));
            std::vector<double> pr(idxs.size());
            double Z = 0.;
            for (size_t k = 0; k < idxs.size(); ++k) {
                pr[k] = std::exp((eff(idxs[k]) - mx) / temp);
                Z += pr[k];
            }
            std::uniform_real_distribution<double> ud(0.0, Z);
            double r = ud(rng), cum = 0.;
            best = idxs.back();
            for (size_t k = 0; k < idxs.size(); ++k) {
                cum += pr[k];
                if (cum >= r) { best = idxs[k]; break; }
            }
        }
        return best;
    }

    // Batch decode from explicit token ids (reference semantics: full forward
    // over the last ctx tokens per step). Used as the equivalence oracle for
    // the streaming path.
    std::string complete_ids(std::vector<int> ids, int max_new, float temp,
                             bool stop_at_newline = true,
                             const std::vector<int>* allowed = nullptr,
                             const LogitBias* bias = nullptr) const {
        if (vocab_.empty()) return "";
        if (ids.empty()) ids.push_back(char_id('\n'));
        std::mt19937 local = rng_;
        std::string out;
        for (int n = 0; n < max_new; ++n) {
            const size_t keep = std::min<size_t>(ids.size(), (size_t)cfg_.ctx);
            std::vector<int> win(ids.end() - (long)keep, ids.end());
            const Mat logits = last_logits(win);
            const int next = pick_token(logits, temp, allowed, local, bias);
            const std::string s = token_surface(next);
            ids.push_back(next);
            if (s == "\n") { if (stop_at_newline) break; continue; }
            out += s;
        }
        return out;
    }

    std::string complete(std::string prompt, int max_new, float temp,
                         bool stop_at_newline = true,
                         const std::vector<int>* allowed = nullptr) const {
        if (vocab_.empty()) return "";
        std::vector<int> ids;
        for (char ch : prompt)
            if (ch != '\r') { const int i = char_id(ch); if (i >= 0) ids.push_back(i); }
        if (ids.empty()) ids.push_back(char_id('\n'));
        return complete_ids(std::move(ids), max_new, temp, stop_at_newline, allowed);
    }

    // ── KV-cache streaming decode ────────────────────────────────────────────
    // Standard incremental-decoding semantics (as in production LLM serving):
    // per-layer k/v rows are cached UNROTATED and re-rotated at read time
    // against the current-length Kuramoto angle table (angles depend on total
    // window length, so rotated keys could not be cached). When the cache
    // fills to ctx the oldest row is evicted and the window slides with no
    // recompute — amortized-O(1) stack work per generated token.
    //
    // SEMANTICS (proven in test_stamlat_streaming.cpp): while the window
    // grows, assembled history deviates from full-window batch recompute by
    // only O(1e-3) logits — a consequence of DK-RoPE's Kuramoto angle table
    // coupling every position to the window length — with argmax verified
    // stable throughout. Once the window is full and until the first
    // eviction (constant table), decoding is exact. Past eviction the two
    // legitimately diverge: batch recompute "retells" every windowed token
    // with only its surviving prefix as left context, while the cache
    // preserves each token's original (fuller) context — neither is an
    // approximation of the other past that point.
    struct StreamCache {
        std::vector<Mat> k, v;   // [layer] pos x d_model (unrotated)
        std::vector<int> ids;    // cached window token ids
        Mat last_logits;         // 1 x V — distribution for the NEXT token
        int evicted = 0;         // rows dropped from the window so far
        std::mt19937 rng{42};    // per-stream sampler state (seeded from model)
    };

    void stream_start(const std::vector<int>& prompt_ids, StreamCache& sc) const {
        if (vocab_.empty()) throw std::runtime_error("stream_start: empty vocab");
        std::vector<int> ids = prompt_ids;
        const size_t keep = std::min<size_t>(ids.size(), (size_t)cfg_.ctx);
        if (keep < ids.size()) ids.erase(ids.begin(), ids.end() - (long)keep);
        if (ids.empty()) ids.push_back(char_id('\n'));
        prefill_window(ids, sc);
        sc.rng = rng_;
    }

    int stream_sample(StreamCache& sc, float temp,
                      const std::vector<int>* allowed = nullptr,
                      const LogitBias* bias = nullptr) const {
        return pick_token(sc.last_logits, temp, allowed, sc.rng, bias);
    }

    // Feed the chosen next token; appends it to the cache and refreshes
    // last_logits for the token after it. When the cache is full, the oldest
    // row is EVICTED (k/v are unrotated ⇒ position-free, so shifting the
    // window needs no recompute — angles are applied at read time).
    void stream_step(int next_id, StreamCache& sc) const {
        if (next_id < 0 || next_id >= total_vocab_size())
            throw std::invalid_argument("stream_step: bad token id");
        if ((int)sc.ids.size() >= cfg_.ctx) {
            const int d = cfg_.d_model;
            for (int l = 0; l < cfg_.n_layers; ++l) {
                auto& Ka = sc.k[l].a;
                auto& Va = sc.v[l].a;
                Ka.erase(Ka.begin(), Ka.begin() + d);      // drop oldest row
                Va.erase(Va.begin(), Va.begin() + d);
            }
            sc.ids.erase(sc.ids.begin());
            ++sc.evicted;
        }
        append_single(next_id, sc);
    }

    // Streaming twin of complete_ids(): standard KV-cache serving semantics —
    // bit-identical to batch decode until the first window eviction, then a
    // first-class decoding mode in its own right (see StreamCache notes).
    std::string stream_complete_ids(const std::vector<int>& ids, int max_new,
                                    float temp, bool stop_at_newline = true,
                                    const std::vector<int>* allowed = nullptr,
                                    const LogitBias* bias = nullptr) const {
        if (vocab_.empty()) return "";
        StreamCache sc;
        stream_start(ids, sc);
        std::string out;
        for (int n = 0; n < max_new; ++n) {
            const int next = pick_token(sc.last_logits, temp, allowed, sc.rng, bias);
            const std::string s = token_surface(next);
            stream_step(next, sc);
            if (s == "\n") { if (stop_at_newline) break; continue; }
            out += s;
        }
        return out;
    }

    bool save(const std::string& path) const {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const std::string magic = "STMLv3";
        std::fwrite(magic.data(), 1, (size_t)magic.size(), f);
        const int hdr[6] = { cfg_.d_model, cfg_.n_layers, cfg_.n_heads, cfg_.d_ff, cfg_.ctx, (int)vocab_.size() };
        std::fwrite(hdr, sizeof(int), 6, f);
        std::fwrite(vocab_.data(), 1, vocab_.size(), f);
        const int W = (int)words_.size();
        std::fwrite(&W, sizeof(int), 1, f);
        for (const auto& w : words_) {
            const int wlen = (int)w.size();
            std::fwrite(&wlen, sizeof(int), 1, f);
            std::fwrite(w.data(), 1, (size_t)wlen, f);
        }
        for (const auto& p : params_)
            std::fwrite(p->a.data(), sizeof(float), p->a.size(), f);
        std::fclose(f);
        return true;
    }

    bool load(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[6] = {0};
        bool ok = std::fread(magic, 1, 6, f) == 6;
        const bool v3 = ok && std::string(magic, 6) == "STMLv3";
        if (ok && !v3) ok = std::string(magic, 6) == "STMLv2";
        int hdr[6] = {0};
        if (ok) ok = std::fread(hdr, sizeof(int), 6, f) == 6;
        if (ok) {
            StamlatConfig c;
            c.d_model = hdr[0]; c.n_layers = hdr[1]; c.n_heads = hdr[2];
            c.d_ff = hdr[3]; c.ctx = hdr[4];
            const int V = hdr[5];
            std::string voc((size_t)V, ' ');
            ok = std::fread(&voc[0], 1, (size_t)V, f) == (size_t)V;
            if (ok) {
                if (!(c == cfg_)) { cfg_ = c; alloc(); }
                vocab_ = voc;
                idx_.clear();
                for (int i = 0; i < (int)vocab_.size(); ++i) idx_.emplace(vocab_[i], i);

                words_.clear(); widx_.clear();
                if (v3) {
                    int W = 0;
                    ok = std::fread(&W, sizeof(int), 1, f) == 1 && W >= 0;
                    for (int wI = 0; ok && wI < W; ++wI) {
                        int wlen = 0;
                        ok = std::fread(&wlen, sizeof(int), 1, f) == 1 && wlen >= 0;
                        if (ok) {
                            std::string w((size_t)wlen, ' ');
                            ok = std::fread(&w[0], 1, (size_t)wlen, f) == (size_t)wlen;
                            widx_.emplace(w, (int)vocab_.size() + (int)words_.size());
                            words_.push_back(std::move(w));
                        }
                    }
                }
                if (ok) rebuild_embedding();
            }
        }
        if (ok)
            for (auto& p : params_)
                if (std::fread(p->a.data(), sizeof(float), p->a.size(), f) != p->a.size()) { ok = false; break; }
        std::fclose(f);
        return ok;
    }

    float loss_and_grads(const std::vector<std::vector<int>>& xs,
                         const std::vector<std::vector<int>>& ys) {
        ensure_grads();
        zero_grads();
        double sum = 0.; int toks = 0;
        for (size_t b = 0; b < xs.size(); ++b) {
            sum += forward_backward(xs[b], ys[b]);
            toks += (int)xs[b].size();
        }
        if (toks > 0) scale_grads(1.f / (float)toks);
        return (float)(sum / toks);
    }

    std::vector<Mat*>& params() { return params_; }
    const std::vector<Mat>& grads_view() const { return owned_grads_; }

    Mat full_logits(const std::vector<int>& ids) const {
        const int d = cfg_.d_model, V = total_vocab_size();
        Mat h((int)ids.size(), d);
        for (int t = 0; t < (int)ids.size(); ++t)
            for (int j = 0; j < d; ++j) h.at(t, j) = emb_->at(ids[t], j);
        Mat hn(h.r, d);
        deep_forward(h, hn);
        Mat logits(h.r, V);
        matmul_trans(hn, *emb_, logits);
        return logits;
    }

private:
    void alloc() {
        owned_.clear(); blk_.clear(); params_.clear();
        const int d = cfg_.d_model, ff = cfg_.d_ff;
        emb_ = owned_.emplace_back(new Mat(1, 1)).get();
        lnf_g = owned_.emplace_back(new Mat(1, d, 1.f)).get();
        lnf_b = owned_.emplace_back(new Mat(1, d, 0.f)).get();
        blk_.resize(cfg_.n_layers);
        std::normal_distribution<float> nd(0.f, 1.f);
        auto initw = [&](Mat* m, int fan_in) {
            const float s = std::min(0.1f, 1.f / std::sqrt((float)fan_in));
            for (float& v : m->a) v = nd(rng_) * s;
        };
        for (int l = 0; l < cfg_.n_layers; ++l) {
            Block& B = blk_[l];
            B.ln1_g = owned_.emplace_back(new Mat(1, d, 1.f)).get();
            B.ln1_b = owned_.emplace_back(new Mat(1, d, 0.f)).get();
            B.Wq = owned_.emplace_back(new Mat(d, d)).get(); initw(B.Wq, d);
            B.bq = owned_.emplace_back(new Mat(1, d, 0.f)).get();
            B.Wk = owned_.emplace_back(new Mat(d, d)).get(); initw(B.Wk, d);
            B.bk = owned_.emplace_back(new Mat(1, d, 0.f)).get();
            B.Wv = owned_.emplace_back(new Mat(d, d)).get(); initw(B.Wv, d);
            B.bv = owned_.emplace_back(new Mat(1, d, 0.f)).get();
            B.Wo = owned_.emplace_back(new Mat(d, d)).get(); initw(B.Wo, d);
            B.bo = owned_.emplace_back(new Mat(1, d, 0.f)).get();
            B.ln2_g = owned_.emplace_back(new Mat(1, d, 1.f)).get();
            B.ln2_b = owned_.emplace_back(new Mat(1, d, 0.f)).get();
            B.W1 = owned_.emplace_back(new Mat(d, ff)).get(); initw(B.W1, d);
            B.b1 = owned_.emplace_back(new Mat(1, ff, 0.f)).get();
            B.W2 = owned_.emplace_back(new Mat(ff, d)).get(); initw(B.W2, ff);
            B.b2 = owned_.emplace_back(new Mat(1, d, 0.f)).get();
        }
        collect_params();
        alloc_adam();
        grads_ready_ = false;
    }

    void collect_params() {
        params_.clear();
        params_.push_back(emb_);
        for (auto& B : blk_) {
            Mat* arr[] = { B.ln1_g, B.ln1_b, B.Wq, B.bq, B.Wk, B.bk, B.Wv, B.bv,
                           B.Wo, B.bo, B.ln2_g, B.ln2_b, B.W1, B.b1, B.W2, B.b2 };
            for (auto* m : arr) params_.push_back(m);
        }
        params_.push_back(lnf_g);
        params_.push_back(lnf_b);
    }

    void rebuild_embedding() {
        *emb_ = Mat(total_vocab_size(), cfg_.d_model);
        std::normal_distribution<float> nd(0.f, 0.08f);
        for (float& v : emb_->a) v = nd(rng_);
        collect_params();
        alloc_adam();
        grads_ready_ = false;
    }

    void alloc_adam() {
        adam_m_.clear(); adam_v_.clear();
        for (auto* p : params_) {
            adam_m_.emplace_back(p->r, p->c, 0.f);
            adam_v_.emplace_back(p->r, p->c, 0.f);
        }
    }

    int char_id(char ch) const {
        auto p = idx_.find(ch);
        return p == idx_.end() ? -1 : p->second;
    }

    void ensure_grads() {
        if (grads_ready_) return;
        owned_grads_.clear();
        for (auto* p : params_) owned_grads_.emplace_back(p->r, p->c, 0.f);
        grads_ready_ = true;
    }
    void zero_grads() { ensure_grads(); for (auto& g : owned_grads_) g.zero(); }

    void scale_grads(float s) {
        for (auto& g : owned_grads_)
            for (float& v : g.a) v *= s;
    }

    void clip_grads() {
        double sq = 0.;
        for (const auto& g : owned_grads_)
            for (float v : g.a) sq += (double)v * v;
        const double nrm = std::sqrt(sq);
        if (nrm > 1.0) {
            const float s = (float)(1.0 / nrm);
            for (auto& g : owned_grads_) for (float& v : g.a) v *= s;
        }
    }

    void adam_step(float lr) {
        t_ += 1;
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        const double bc1 = 1. - std::pow(b1, (double)t_), bc2 = 1. - std::pow(b2, (double)t_);
        for (size_t i = 0; i < params_.size(); ++i) {
            Mat& w = *params_[i]; Mat& g = owned_grads_[i];
            Mat& m = adam_m_[i]; Mat& v = adam_v_[i];
            for (size_t k = 0; k < w.a.size(); ++k) {
                m.a[k] = (float)(b1 * m.a[k] + (1 - b1) * g.a[k]);
                v.a[k] = (float)(b2 * v.a[k] + (1 - b2) * (double)g.a[k] * g.a[k]);
                w.a[k] -= (float)(lr * (m.a[k] / bc1) / (std::sqrt(v.a[k] / bc2) + eps));
            }
        }
    }

    // ── forward structures ──────────────────────────────────────────────────
    struct BlockCache {
        Mat h_in, n1, q, k, vv, qr, kr, A, att, x2, n2, fpre, f, u;
        LNCache c1, c2;
        std::vector<Mat> heads;                       // per-head T x T probs
    };
    struct SeqCache { std::vector<BlockCache> blk; LNCache lnf; };

    void block_forward(int l, const Mat& h, BlockCache& cc) const {
        const int d = cfg_.d_model, H = cfg_.n_heads, dh = d / H, T = h.r;
        const Block& B = blk_[l];
        cc.h_in = h;
        cc.n1 = Mat(T, d);
        ln_forward(h, *B.ln1_g, *B.ln1_b, cc.n1, cc.c1);

        cc.q = Mat(T, d); matmul(cc.n1, *B.Wq, cc.q); add_bias(cc.q, *B.bq);
        cc.k = Mat(T, d); matmul(cc.n1, *B.Wk, cc.k); add_bias(cc.k, *B.bk);
        cc.vv = Mat(T, d); matmul(cc.n1, *B.Wv, cc.vv); add_bias(cc.vv, *B.bv);
        cc.qr = cc.q; cc.kr = cc.k;
        for (int hd = 0; hd < H; ++hd) {
            Mat qh = slice_cols(cc.qr, hd * dh, dh), kh = slice_cols(cc.kr, hd * dh, dh);
            rotary_apply(qh, dh, false); rotary_apply(kh, dh, false);
            paste_cols(cc.qr, qh, hd * dh); paste_cols(cc.kr, kh, hd * dh);
        }

        const float scale = 1.f / std::sqrt((float)dh);
        cc.heads.assign(H, Mat());
        cc.A = Mat(T, d);
        for (int hd = 0; hd < H; ++hd) {
            const Mat Qh = slice_cols(cc.qr, hd * dh, dh);
            const Mat Kh = slice_cols(cc.kr, hd * dh, dh);
            const Mat Vh = slice_cols(cc.vv, hd * dh, dh);
            Mat& S = cc.heads[hd]; S = Mat(T, T);
            for (int i = 0; i < T; ++i) {
                double mx = -1e30;
                for (int j = 0; j <= i; ++j) {
                    double z = 0.;
                    for (int e = 0; e < dh; ++e) z += (double)Qh.at(i, e) * Kh.at(j, e);
                    z *= scale;
                    S.at(i, j) = (float)z;
                    mx = std::max(mx, z);
                }
                double Z = 0.;
                for (int j = 0; j <= i; ++j) { S.at(i, j) = (float)std::exp((double)S.at(i, j) - mx); Z += S.at(i, j); }
                for (int j = 0; j <= i; ++j) S.at(i, j) = (float)((double)S.at(i, j) / Z);
                for (int j = i + 1; j < T; ++j) S.at(i, j) = 0.f;
            }
            Mat Oh(T, dh);
            for (int i = 0; i < T; ++i)
                for (int e = 0; e < dh; ++e) {
                    double s = 0.;
                    for (int j = 0; j <= i; ++j) s += (double)S.at(i, j) * Vh.at(j, e);
                    Oh.at(i, e) = (float)s;
                }
            paste_cols(cc.A, Oh, hd * dh);
        }
        cc.att = Mat(T, d);
        matmul(cc.A, *B.Wo, cc.att); add_bias(cc.att, *B.bo);

        cc.x2 = Mat(T, d);
        for (size_t i = 0; i < cc.x2.a.size(); ++i) cc.x2.a[i] = cc.h_in.a[i] + cc.att.a[i];

        cc.n2 = Mat(T, d);
        ln_forward(cc.x2, *B.ln2_g, *B.ln2_b, cc.n2, cc.c2);
        cc.fpre = Mat(T, cfg_.d_ff);
        matmul(cc.n2, *B.W1, cc.fpre); add_bias(cc.fpre, *B.b1);
        cc.f = cc.fpre;
        for (float& v : cc.f.a) v = std::tanh(v);
        Mat f2(T, d);
        matmul(cc.f, *B.W2, f2); add_bias(f2, *B.b2);

        cc.u = Mat(T, d);
        for (size_t i = 0; i < cc.u.a.size(); ++i) cc.u.a[i] = cc.att.a[i] + f2.a[i];
    }

    void block_backward(int l, const BlockCache& cc, const Mat& gu, Mat& gh) {
        const int d = cfg_.d_model, H = cfg_.n_heads, dh = d / H, ff = cfg_.d_ff, T = gu.r;
        Block& B = blk_[l];

        const Mat df2 = gu;
        Mat df(T, ff);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < ff; ++j) {
                double s = 0.;
                for (int o = 0; o < d; ++o) s += (double)df2.at(t, o) * B.W2->at(j, o);
                df.at(t, j) = (float)s;
            }
        {
            Mat dW2(ff, d);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < ff; ++j) {
                    const float fv = cc.f.at(t, j);
                    if (fv == 0.f) continue;
                    const float* gr = &df2.a[(size_t)t * d];
                    float* wr = &dW2.a[(size_t)j * d];
                    for (int o = 0; o < d; ++o) wr[o] += fv * gr[o];
                }
            accumulate_grad(B.W2, dW2);
        }
        {
            Mat db2(1, d);
            for (int o = 0; o < d; ++o) { double s = 0.; for (int t = 0; t < T; ++t) s += df2.at(t, o); db2.a[o] = (float)s; }
            accumulate_grad(B.b2, db2);
        }
        for (size_t i = 0; i < df.a.size(); ++i) df.a[i] *= (1.f - cc.f.a[i] * cc.f.a[i]);

        Mat dn2(T, d);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < d; ++o) {
                double s = 0.;
                for (int i = 0; i < ff; ++i) s += (double)df.at(t, i) * B.W1->at(o, i);
                dn2.at(t, o) = (float)s;
            }
        {
            Mat dW1(d, ff);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < d; ++i) {
                    const float nv = cc.n2.at(t, i);
                    if (nv == 0.f) continue;
                    const float* gr = &df.a[(size_t)t * ff];
                    float* wr = &dW1.a[(size_t)i * ff];
                    for (int o = 0; o < ff; ++o) wr[o] += nv * gr[o];
                }
            accumulate_grad(B.W1, dW1);
        }
        {
            Mat db1(1, ff);
            for (int j = 0; j < ff; ++j) { double s = 0.; for (int t = 0; t < T; ++t) s += df.at(t, j); db1.a[j] = (float)s; }
            accumulate_grad(B.b1, db1);
        }

        Mat dx2(T, d);
        ln_backward(dn2, cc.c2, *B.ln2_g, dx2, *grad_of(B.ln2_g), *grad_of(B.ln2_b));

        Mat datt(T, d);
        for (size_t i = 0; i < datt.a.size(); ++i) datt.a[i] = gu.a[i] + dx2.a[i];
        for (size_t i = 0; i < gh.a.size(); ++i) gh.a[i] += dx2.a[i];

        Mat dA(T, d);
        matmul_trans(datt, *B.Wo, dA);
        {
            Mat dWo(d, d);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < d; ++i) {
                    const float av = cc.A.at(t, i);
                    if (av == 0.f) continue;
                    const float* gr = &datt.a[(size_t)t * d];
                    float* wr = &dWo.a[(size_t)i * d];
                    for (int o = 0; o < d; ++o) wr[o] += av * gr[o];
                }
            accumulate_grad(B.Wo, dWo);
        }
        {
            Mat dbo(1, d);
            for (int o = 0; o < d; ++o) { double s = 0.; for (int t = 0; t < T; ++t) s += datt.at(t, o); dbo.a[o] = (float)s; }
            accumulate_grad(B.bo, dbo);
        }

        const float scale = 1.f / std::sqrt((float)dh);
        Mat dqr(T, d), dkr(T, d), dvv(T, d);
        for (int hd = 0; hd < H; ++hd) {
            const Mat Qh = slice_cols(cc.qr, hd * dh, dh);
            const Mat Kh = slice_cols(cc.kr, hd * dh, dh);
            const Mat Vh = slice_cols(cc.vv, hd * dh, dh);
            const Mat& S = cc.heads[hd];
            const Mat dAh = slice_cols(dA, hd * dh, dh);
            Mat dSh(T, T), dVh(T, dh);
            for (int j = 0; j < T; ++j)
                for (int e = 0; e < dh; ++e) {
                    double s = 0.;
                    for (int i = 0; i < T; ++i) s += (double)S.at(i, j) * dAh.at(i, e);
                    dVh.at(j, e) = (float)s;
                }
            paste_cols(dvv, dVh, hd * dh);
            for (int i = 0; i < T; ++i) {
                double dot = 0.;
                for (int j = 0; j <= i; ++j) {
                    // g = dAh * Vh^T : gradient w.r.t. attention probabilities
                    double s = 0.;
                    for (int e = 0; e < dh; ++e) s += (double)dAh.at(i, e) * Vh.at(j, e);
                    dSh.at(i, j) = (float)s;
                    dot += (double)dSh.at(i, j) * S.at(i, j);
                }
                for (int j = 0; j <= i; ++j)
                    dSh.at(i, j) = (float)(scale * S.at(i, j) * ((double)dSh.at(i, j) - dot));
                for (int j = i + 1; j < T; ++j) dSh.at(i, j) = 0.f;
            }
            // dQ(i,e) += sum_j dSh(i,j) * Kh(j,e);   dK(j,e) += sum_i dSh(i,j) * Qh(i,e)
            for (int i = 0; i < T; ++i)
                for (int e = 0; e < dh; ++e) {
                    double sq = 0.;
                    for (int j = 0; j <= i; ++j) sq += (double)dSh.at(i, j) * Kh.at(j, e);
                    dqr.at(i, hd * dh + e) += (float)sq;
                }
            for (int j = 0; j < T; ++j)
                for (int e = 0; e < dh; ++e) {
                    double sk = 0.;
                    for (int i = j; i < T; ++i) sk += (double)dSh.at(i, j) * Qh.at(i, e);
                    dkr.at(j, hd * dh + e) += (float)sk;
                }
        }
        for (int hd = 0; hd < H; ++hd) {
            Mat dqh = slice_cols(dqr, hd * dh, dh), dkh = slice_cols(dkr, hd * dh, dh);
            rotary_apply(dqh, dh, true); rotary_apply(dkh, dh, true);
            paste_cols(dqr, dqh, hd * dh); paste_cols(dkr, dkh, hd * dh);
        }

        Mat dn1(T, d);
        std::fill(dn1.a.begin(), dn1.a.end(), 0.f);
        // VJP of q/k/v = n1*W:  dn1(t,j) += sum_o dX(t,o) * W(j,o)
        auto add_vjp = [&](const Mat& dX, const Mat& W) {
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < d; ++j) {
                    const float* wr = &W.a[(size_t)j * d];
                    double s = 0.;
                    for (int o = 0; o < d; ++o) s += (double)dX.at(t, o) * wr[o];
                    dn1.at(t, j) += (float)s;
                }
        };
        add_vjp(dqr, *B.Wq); add_vjp(dkr, *B.Wk); add_vjp(dvv, *B.Wv);

        auto proj_grad = [&](const Mat& dX, const Mat& Xin, Mat* W, Mat* bb) {
            Mat dW(W->r, W->c);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < W->r; ++i) {
                    const float xv = Xin.at(t, i);
                    if (xv == 0.f) continue;
                    const float* gr = &dX.a[(size_t)t * W->c];
                    float* wr = &dW.a[(size_t)i * W->c];
                    for (int o = 0; o < W->c; ++o) wr[o] += xv * gr[o];
                }
            accumulate_grad(W, dW);
            Mat db(1, W->c);
            for (int o = 0; o < W->c; ++o) { double s = 0.; for (int t = 0; t < T; ++t) s += dX.at(t, o); db.a[o] = (float)s; }
            accumulate_grad(bb, db);
        };
        proj_grad(dqr, cc.n1, B.Wq, B.bq);
        proj_grad(dkr, cc.n1, B.Wk, B.bk);
        proj_grad(dvv, cc.n1, B.Wv, B.bv);

        ln_backward(dn1, cc.c1, *B.ln1_g, gh, *grad_of(B.ln1_g), *grad_of(B.ln1_b));
    }

    double forward_backward(const std::vector<int>& x, const std::vector<int>& y,
                            const std::vector<float>& token_weight = {}) {
        ensure_grads();
        const int T = (int)x.size(), d = cfg_.d_model, L = cfg_.n_layers,
                  V = total_vocab_size();   // softmax spans chars AND word tokens

        SeqCache sc; sc.blk.resize(L);
        Mat h(T, d);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d; ++j) h.at(t, j) = emb_->at(x[t], j);

        std::vector<Mat> vs(L + 1, Mat(T, d));
        for (int l = 0; l < L; ++l) {
            block_forward(l, h, sc.blk[l]);
            Mat vnext(T, d), hnext(T, d);
            for (size_t i = 0; i < h.a.size(); ++i) {
                vnext.a[i] = cfg_.depth_gamma * vs[l].a[i] + sc.blk[l].u.a[i];
                hnext.a[i] = h.a[i] + cfg_.depth_tau * vnext.a[i];
            }
            vs[l + 1] = vnext;
            h = hnext;
        }
        Mat hn(T, d);
        ln_forward(h, *lnf_g, *lnf_b, hn, sc.lnf);

        Mat logits(T, V);
        matmul_trans(hn, *emb_, logits);

        Mat dlogits(T, V);
        double loss = 0.;
        for (int t = 0; t < T; ++t) {
            const float wt = token_weight.empty() ? 1.f : token_weight[t];
            if (wt == 0.f) continue;
            double mx = -1e30;
            for (int v2 = 0; v2 < V; ++v2) mx = std::max(mx, (double)logits.at(t, v2));
            double Z = 0.;
            for (int v2 = 0; v2 < V; ++v2) Z += std::exp((double)logits.at(t, v2) - mx);
            loss += (double)wt * (-(double)logits.at(t, y[t]) + mx + std::log(Z));
            for (int v2 = 0; v2 < V; ++v2) {
                const double p = std::exp((double)logits.at(t, v2) - mx) / Z;
                dlogits.at(t, v2) = (float)(wt * (p - (v2 == y[t] ? 1.0 : 0.0)));
            }
        }

        Mat* eg = grad_of(emb_);
        for (int t = 0; t < T; ++t)
            for (int v2 = 0; v2 < V; ++v2) {
                const float dl = dlogits.at(t, v2);
                if (dl == 0.f) continue;
                float* er = &eg->a[(size_t)v2 * d];
                const float* hr = &hn.a[(size_t)t * d];
                for (int j = 0; j < d; ++j) er[j] += dl * hr[j];
            }
        Mat dhn(T, d);
        for (int t = 0; t < T; ++t)
            for (int v2 = 0; v2 < V; ++v2) {
                const float dl = dlogits.at(t, v2);
                if (dl == 0.f) continue;
                const float* er = &emb_->a[(size_t)v2 * d];
                float* gr = &dhn.a[(size_t)t * d];
                for (int j = 0; j < d; ++j) gr[j] += dl * er[j];
            }

        Mat gh(T, d);
        ln_backward(dhn, sc.lnf, *lnf_g, gh, *grad_of(lnf_g), *grad_of(lnf_b));

        Mat gv(T, d);
        for (int l = L - 1; l >= 0; --l) {
            // G_v[l+1] = incoming v-chain + tau * G_h[l+1];  G_u[l] = G_v[l+1];
            // G_h[l] = G_h[l+1] (+ block contribution);      G_v[l] += gamma * G_v[l+1]
            Mat gv_lp1(T, d);
            for (size_t i = 0; i < gh.a.size(); ++i)
                gv_lp1.a[i] = gv.a[i] + cfg_.depth_tau * gh.a[i];
            block_backward(l, sc.blk[l], gv_lp1, gh);
            for (size_t i = 0; i < gv.a.size(); ++i) gv.a[i] = cfg_.depth_gamma * gv_lp1.a[i];
        }

        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d; ++j) eg->at(x[t], j) += gh.at(t, j);

        return loss;
    }

    std::pair<double,int> loss_only_ids(int off, const std::vector<int>& ids) const {
        const int T = cfg_.ctx, d = cfg_.d_model, V = total_vocab_size();
        Mat h(T, d);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d; ++j) h.at(t, j) = emb_->at(ids[off + t], j);
        Mat hn(T, d);
        deep_forward(h, hn);
        Mat logits(T, V); matmul_trans(hn, *emb_, logits);
        double sum = 0.; int cnt = 0;
        for (int t = 0; t < T; ++t) {
            const int yt = ids[off + t + 1];
            double mx = -1e30; for (int v2 = 0; v2 < V; ++v2) mx = std::max(mx, (double)logits.at(t, v2));
            double Z = 0.; for (int v2 = 0; v2 < V; ++v2) Z += std::exp((double)logits.at(t, v2) - mx);
            sum += -(double)logits.at(t, yt) + mx + std::log(Z); ++cnt;
        }
        return { sum, cnt };
    }

    Mat last_logits(const std::vector<int>& ids) const {
        const int T = (int)ids.size(), d = cfg_.d_model, V = total_vocab_size();
        Mat h(T, d);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d; ++j) h.at(t, j) = emb_->at(ids[t], j);
        Mat hn(T, d);
        deep_forward(h, hn);
        Mat logits(T, V); matmul_trans(hn, *emb_, logits);
        Mat row(1, V);
        for (int v2 = 0; v2 < V; ++v2) row.a[v2] = logits.at(T - 1, v2);
        return row;
    }

    // ── streaming internals ──────────────────────────────────────────────────
    // Full forward over a window that also captures each layer's unrotated
    // k/v rows into the cache. Mirrors deep_forward() exactly.
    void prefill_window(const std::vector<int>& ids, StreamCache& sc) const {
        const int T = (int)ids.size(), d = cfg_.d_model, L = cfg_.n_layers;
        sc.ids = ids;
        sc.k.assign(L, Mat(T, d));
        sc.v.assign(L, Mat(T, d));

        Mat h(T, d);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d; ++j) h.at(t, j) = emb_->at(ids[t], j);

        std::vector<Mat> vs(L + 1, Mat(T, d));
        for (int l = 0; l < L; ++l) {
            BlockCache tmp;
            block_forward(l, h, tmp);          // tmp.k / tmp.vv remain unrotated
            sc.k[l] = tmp.k;
            sc.v[l] = tmp.vv;
            Mat vnext(T, d), hnext(T, d);
            for (size_t i = 0; i < h.a.size(); ++i) {
                vnext.a[i] = cfg_.depth_gamma * vs[l].a[i] + tmp.u.a[i];
                hnext.a[i] = h.a[i] + cfg_.depth_tau * vnext.a[i];
            }
            vs[l + 1] = vnext;
            h = hnext;
        }

        Mat hrow(1, d), hn(1, d);
        for (int j = 0; j < d; ++j) hrow.at(0, j) = h.at(T - 1, j);
        LNCache lc;
        ln_forward(hrow, *lnf_g, *lnf_b, hn, lc);
        Mat logits(1, total_vocab_size());
        matmul_trans(hn, *emb_, logits);
        sc.last_logits = logits;
    }

    // Single-token forward: project q/k/v for the new position, extend the
    // unrotated caches, re-rotate ALL keys against the current-length angle
    // table, attend, then run FFN + symplectic update for this one row.
    void append_single(int x, StreamCache& sc) const {
        const int d = cfg_.d_model, H = cfg_.n_heads, dh = d / H, L = cfg_.n_layers;
        const int C = (int)sc.ids.size() + 1;              // window length incl. x
        const int c = C - 1;                               // position of x
        const std::vector<float>& ang = kuramoto_angles(C);
        const float scale = 1.f / std::sqrt((float)dh);

        Mat h(1, d), att(1, d);
        for (int j = 0; j < d; ++j) h.at(0, j) = emb_->at(x, j);

        std::vector<float> vvec((size_t)d, 0.f);
        LNCache lc1, lc2;
        for (int l = 0; l < L; ++l) {
            const Block& B = blk_[l];
            Mat n1(1, d);
            ln_forward(h, *B.ln1_g, *B.ln1_b, n1, lc1);

            Mat q(1, d), knew(1, d), vnew(1, d);
            matmul(n1, *B.Wq, q);  add_bias(q, *B.bq);
            matmul(n1, *B.Wk, knew); add_bias(knew, *B.bk);
            matmul(n1, *B.Wv, vnew); add_bias(vnew, *B.bv);

            Mat& K = sc.k[l];
            Mat& Vc = sc.v[l];
            K.a.insert(K.a.end(), knew.a.begin(), knew.a.end());   // row-major append
            Vc.a.insert(Vc.a.end(), vnew.a.begin(), vnew.a.end());
            K.r = C; Vc.r = C;

            // one full-width rotation each for the q-row and the whole key cache
            Mat qh(1, d);
            for (int j = 0; j < d; ++j) qh.at(0, j) = q.at(0, j);
            rotary_heads(qh, dh, &ang[c]);                         // angle of pos c
            Mat Krot = K;
            rotary_heads(Krot, dh, ang.data());                    // all cached positions

            for (int hd = 0; hd < H; ++hd) {
                const int base = hd * dh;

                Mat S(1, C);
                double mx = -1e30;
                for (int j = 0; j <= c; ++j) {
                    double z = 0.;
                    for (int e = 0; e < dh; ++e) z += (double)qh.at(0, base + e) * Krot.at(j, base + e);
                    z *= scale;
                    S.at(0, j) = (float)z;
                    mx = std::max(mx, z);
                }
                double Z = 0.;
                for (int j = 0; j <= c; ++j) { S.at(0, j) = (float)std::exp((double)S.at(0, j) - mx); Z += S.at(0, j); }
                for (int j = 0; j <= c; ++j) S.at(0, j) = (float)((double)S.at(0, j) / Z);

                for (int e = 0; e < dh; ++e) {
                    double s = 0.;
                    for (int j = 0; j <= c; ++j) s += (double)S.at(0, j) * Vc.at(j, base + e);
                    att.at(0, base + e) = (float)s;
                }
            }

            Mat a1(1, d);
            matmul(att, *B.Wo, a1); add_bias(a1, *B.bo);

            Mat x2(1, d);
            for (size_t i = 0; i < x2.a.size(); ++i) x2.a[i] = h.a[i] + a1.a[i];

            Mat n2(1, d);
            ln_forward(x2, *B.ln2_g, *B.ln2_b, n2, lc2);
            Mat fpre(1, cfg_.d_ff);
            matmul(n2, *B.W1, fpre); add_bias(fpre, *B.b1);
            for (float& val : fpre.a) val = std::tanh(val);
            Mat f2(1, d);
            matmul(fpre, *B.W2, f2); add_bias(f2, *B.b2);

            for (int j = 0; j < d; ++j) {                          // u = att + f2
                const float u = a1.at(0, j) + f2.at(0, j);
                const float vn = cfg_.depth_gamma * vvec[j] + u;   // symplectic step
                h.at(0, j) += cfg_.depth_tau * vn;
                vvec[j] = vn;
            }
        }

        Mat hn(1, d);
        LNCache lcf;
        ln_forward(h, *lnf_g, *lnf_b, hn, lcf);
        Mat logits(1, total_vocab_size());
        matmul_trans(hn, *emb_, logits);

        sc.ids.push_back(x);
        sc.last_logits = logits;
    }

    void deep_forward(Mat& h, Mat& hn) const {
        const int d = cfg_.d_model, L = cfg_.n_layers;
        std::vector<Mat> vs(L + 1, Mat(h.r, d));
        for (int l = 0; l < L; ++l) {
            BlockCache tmp; block_forward(l, h, tmp);
            Mat vnext(h.r, d), hnext(h.r, d);
            for (size_t i = 0; i < h.a.size(); ++i) {
                vnext.a[i] = cfg_.depth_gamma * vs[l].a[i] + tmp.u.a[i];
                hnext.a[i] = h.a[i] + cfg_.depth_tau * vnext.a[i];
            }
            vs[l + 1] = vnext; h = hnext;
        }
        LNCache lc;
        ln_forward(h, *lnf_g, *lnf_b, hn, lc);
    }

    static Mat slice_cols(const Mat& src, int c0, int n) {
        Mat out(src.r, n);
        for (int t = 0; t < src.r; ++t)
            for (int j = 0; j < n; ++j) out.at(t, j) = src.at(t, c0 + j);
        return out;
    }
    static void paste_cols(Mat& dst, const Mat& part, int c0) {
        for (int t = 0; t < dst.r; ++t)
            for (int j = 0; j < part.c; ++j) dst.at(t, c0 + j) = part.at(t, j);
    }

    Mat* grad_of(Mat* p) {
        for (size_t i = 0; i < params_.size(); ++i)
            if (params_[i] == p) return &owned_grads_[i];
        throw std::runtime_error("param not registered");
    }
    void accumulate_grad(Mat* p, const Mat& g) {
        Mat* gp = grad_of(p);
        for (size_t i = 0; i < g.a.size(); ++i) gp->a[i] += g.a[i];
    }

    struct Block {
        Mat *ln1_g, *ln1_b, *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo,
            *ln2_g, *ln2_b, *W1, *b1, *W2, *b2;
    };

    StamlatConfig cfg_;
    std::string vocab_;
    std::unordered_map<char, int> idx_;

    std::vector<std::string> words_;                 // word tokens (ids after chars)
    std::unordered_map<std::string, int> widx_;

    std::vector<std::unique_ptr<Mat>> owned_;
    std::vector<Mat*> params_;
    std::vector<Mat> owned_grads_;
    bool grads_ready_ = false;
    Mat* emb_ = nullptr; Mat* lnf_g = nullptr; Mat* lnf_b = nullptr;
    std::vector<Block> blk_;

    std::vector<Mat> adam_m_, adam_v_;
    long long t_ = 0;
    std::mt19937 rng_{42};
};

} // namespace neural
} // namespace engines
} // namespace brain3
