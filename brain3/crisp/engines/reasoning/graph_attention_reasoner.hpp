#pragma once
/**
 * brain3/crisp/engines/reasoning/graph_attention_reasoner.hpp
 *
 * GRAPH-ATTENTION REASONER — multi-hop inference over the knowledge base.
 *
 * COMPLEX-valued embeddings (ComplEx family):
 *   score(h,r,t) = Re( Σ_j (hr+j·i) · (rr+j·i) · (tr−j·ti) )
 * Asymmetry is native — head-role vs tail-role geometry separates without
 * tricks; relation vectors multiply as complex numbers along paths so ANY
 * relation sequence is scoreable (structural generalization by
 * construction). Inverse-relation augmentation retained for coverage.
 *
 * Training: random walks + rejection-sampled negatives + sampled-softmax
 * RANKING loss (gradients vanish once dst outranks negatives — no churn).
 * Chain-rule gradients through complex products, global-norm clip,
 * unit-sphere projection on the full [re|im] row.
 */
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "reasoning_engine.hpp"    // brain2::reasoning::Fact adapter

namespace brain3 {
namespace engines {
namespace reasoning {

class GraphAttentionReasoner {
public:
    struct Edge { int head, rel, tail; };

    int add_entity(const std::string& name) {
        auto it = eid_.find(name);
        if (it != eid_.end()) return it->second;
        int id = (int)entities_.size();
        eid_.emplace(name, id);
        entities_.push_back(name);
        adj_out_.emplace_back();
        return id;
    }
    int add_relation(const std::string& name) {
        auto it = rid_.find(name);
        if (it != rid_.end()) return it->second;
        int id = (int)relations_.size();
        rid_.emplace(name, id);
        relations_.push_back(name);
        return id;
    }
    void add_edge(int head, int rel, int tail) {
        if (head < 0 || head >= (int)entities_.size() ||
            rel < 0 || rel >= (int)relations_.size() ||
            tail < 0 || tail >= (int)entities_.size())
            throw std::out_of_range("add_edge: unknown entity/relation id");
        edges_.push_back({head, rel, tail});
        adj_out_[head].push_back({rel, tail});   // (relation, tail)
    }
    int entity_count() const { return (int)entities_.size(); }
    int edge_count() const { return (int)edges_.size(); }

    void load_from_facts(const std::set<brain2::reasoning::Fact>& facts) {
        for (const auto& f : facts)
            add_edge(add_entity(f.subj), add_relation(f.rel),
                     add_entity(f.obj));
    }

    // ── training ───────────────────────────────────────────────────────────
    struct TrainConfig {
        int steps = 4000;
        int batch = 32;
        int max_path_len = 3;
        int negatives_per_pos = 5;
        double lr = 0.02;
        unsigned seed = 7;
    };

    void train() { train(TrainConfig{}); }

    void train(const TrainConfig& cfg) {
        rng_.seed(cfg.seed);
        const int V = entity_count(), Rn = (int)relations_.size();
        const int RnA = 2 * Rn;                       // + inverse relations
        const size_t d2 = d2_size();

        std::vector<std::vector<std::pair<int,int>>> out(V);
        true_keys_.clear();
        for (const auto& e : edges_) {
            out[e.head].push_back({e.rel, e.tail});
            out[e.tail].push_back({RnA / 2 + e.rel, e.head});   // inverse
            true_keys_.insert(key(e.head, e.rel, e.tail));
            true_keys_.insert(key(e.tail, RnA / 2 + e.rel, e.head));
        }

        std::normal_distribution<double> nd(0.0, std::sqrt(1.0 / dim_));
        E_.resize((size_t)V * d2);
        for (auto& v : E_) v = nd(rng_);
        R_.resize((size_t)RnA * d2);
        for (auto& v : R_) v = nd(rng_);
        mE_.assign(E_.size(), 0.0); vE_.assign(E_.size(), 0.0);
        mR_.assign(R_.size(), 0.0); vR_.assign(R_.size(), 0.0);
        gradE_.assign(E_.size(), 0.0);
        gradR_.assign(R_.size(), 0.0);
        adam_t_ = 0;

        const double tau = 0.35;
        double prev_loss = -1;
        for (int step = 1; step <= cfg.steps; ++step) {
            std::fill(gradE_.begin(), gradE_.end(), 0.0);
            std::fill(gradR_.begin(), gradR_.end(), 0.0);
            double loss_sum = 0.;
            for (int b = 0; b < cfg.batch; ++b) {
                auto walk = sample_walk(out, cfg.max_path_len);
                const auto& pr = walk.first;
                if (pr.empty()) continue;
                const int src = walk.second.front();
                const int dst = walk.second.back();

                std::vector<int> negs;
                for (int k = 0; k < cfg.negatives_per_pos; ++k) {
                    int cand = (int)(rng_() % V);
                    if (cand != dst &&
                        !true_keys_.count(key(src, pr.back(), cand)))
                        negs.push_back(cand);
                }
                if (negs.empty()) continue;
                loss_sum += rank_step(src, pr, dst, negs, tau, RnA);
            }
            clip_grads();
            adam_step(cfg.lr);

            // project full [re|im] rows onto unit spheres
            norm_all(V, RnA, d2);

            if (step % 200 == 0 || step == 1) {
                std::cerr << "[gar] step " << step << "/" << cfg.steps
                          << " loss " << loss_sum / std::max(1, cfg.batch)
                          << (prev_loss > 0 ? "" : "") << "\n";
                prev_loss = loss_sum;
            }
        }
    }

    // ── querying ───────────────────────────────────────────────────────────
    struct Hit { int entity; double mass; };
    struct QueryResult {
        std::vector<Hit> ranked;
        std::vector<std::map<int, double>> stages;
    };

    QueryResult query_stages(int src, const std::vector<int>& rels,
                             double temperature = 1.0) const {
        std::map<int, double> frontier{{src, 1.0}};
        QueryResult qr;
        qr.stages.push_back(frontier);
        for (int rel : rels) {
            std::map<int, double> raw;
            for (const auto& [u, mass] : frontier) {
                const auto& out = adj_out_[u];
                if (out.empty()) continue;
                double mx = -1e30;
                std::vector<double> en(out.size());
                bool any = false;
                for (size_t i = 0; i < out.size(); ++i) {
                    if (rel >= 0 && out[i].first != rel) { en[i] = -1e30; continue; }
                    const Edge& e = edges_[out[i].first];
                    double s = score_triple(u, e.rel, e.tail);
                    en[i] = s; mx = std::max(mx, s); any = true;
                }
                if (!any || !std::isfinite(mx)) continue;
                double Z = 0.;
                for (double& v2 : en) { v2 = std::exp((v2 - mx)/temperature); Z += v2; }
                if (!(Z > 0)) continue;
                for (size_t i = 0; i < out.size(); ++i) {
                    if (en[i] <= -1e29) continue;
                    raw[out[i].second] += mass * (en[i] / Z);
                }
            }
            frontier.swap(raw);
            qr.stages.push_back(frontier);
        }
        qr.ranked.reserve(frontier.size());
        for (const auto& [e, m] : frontier) qr.ranked.push_back({e, m});
        std::sort(qr.ranked.begin(), qr.ranked.end(),
                  [](const Hit& a, const Hit& b){ return a.mass > b.mass; });
        return qr;
    }

    std::vector<Hit> query(int src, const std::vector<int>& rels,
                           double temperature = 1.0) const {
        return query_stages(src, rels, temperature).ranked;
    }

    // compositional path score: complex product of relations along path
    double path_score(int src, const std::vector<int>& rels, int dst) const {
        std::vector<double> cre, cim;
        compose_path(rels, cre, cim);
        return complex_tail_score(src, cre, cim, dst);
    }

    int entity_id(const std::string& n) const {
        auto it = eid_.find(n); return it == eid_.end() ? -1 : it->second;
    }
    int relation_id(const std::string& n) const {
        auto it = rid_.find(n); return it == rid_.end() ? -1 : it->second;
    }
    const std::string& entity_name(int id) const { return entities_[id]; }

    // sleep-kernel checkpointing
    std::vector<std::vector<double>> snapshot_params() const { return {E_, R_}; }
    void restore_params(const std::vector<std::vector<double>>& s) {
        if (s.size() == 2) { E_ = s[0]; R_ = s[1]; }
    }
    bool trained() const { return !E_.empty(); }

    // self-check MRR of true tails among random negatives (chance ~0.05 @20)
    double self_check_mrr(int samples = 30, int candidates = 20) const {
        if (edges_.empty()) return 0.0;
        std::mt19937 g(2024);
        double mrr = 0.; int n = 0;
        for (int k = 0; k < samples; ++k) {
            const Edge& e = edges_[g() % edges_.size()];
            std::vector<int> cands{e.tail};
            while ((int)cands.size() < candidates)
                cands.push_back((int)(g() % entities_.size()));
            std::shuffle(cands.begin(), cands.end(), g);
            int rank = 1;
            for (int cand : cands)
                if (cand != e.tail &&
                    score_triple(e.head, e.rel, cand) > score_triple(e.head, e.rel, e.tail))
                    ++rank;
            mrr += 1.0 / rank; ++n;
        }
        return n ? mrr / n : 0.0;
    }

private:
    static constexpr int dim_ = 128;
    size_t d2_size() const { return (size_t)2 * dim_; }

    uint64_t key(int h, int r, int t) const {
        return (uint64_t)h * 1000003ULL * 1000003ULL +
               (uint64_t)r * 1000003ULL + (uint64_t)t;
    }

    std::pair<std::vector<int>, std::vector<int>>
    sample_walk(const std::vector<std::vector<std::pair<int,int>>>& out, int L) {
        if (out.empty()) return {};
        int cur = (int)(rng_() % out.size());
        std::vector<int> rels, nodes{cur};
        for (int hop = 0; hop < L; ++hop) {
            const auto& o = out[cur];
            if (o.empty()) break;
            auto [r, t] = o[rng_() % o.size()];
            rels.push_back(r); nodes.push_back(t); cur = t;
        }
        if (rels.size() >= 2 && rng_() % 100 < 40) {   // mixed-length supervision
            rels.resize(1); nodes.resize(2);
        }
        return {rels, nodes};
    }

    double complex_tail_score(int src,
                              const std::vector<double>& cre,
                              const std::vector<double>& cim,
                              int dst) const {
        const int d = dim_, d2 = d2_size();
        const double* H = &E_[(size_t)src * d2];
        const double* T = &E_[(size_t)dst * d2];
        double s = 0.;
        for (int j = 0; j < d; ++j) {
            s += cre[j] * (H[j] * T[j] + H[d + j] * T[d + j])
               + cim[j] * (H[d + j] * T[j] - H[j] * T[d + j]);
        }
        return s;
    }

    double score_triple(int h, int r, int t) const {
        const int d = dim_, d2 = d2_size();
        const double* Rr = &R_[(size_t)r * d2];
        const double* H = &E_[(size_t)h * d2];
        const double* T = &E_[(size_t)t * d2];
        double s = 0.;
        for (int j = 0; j < d; ++j) {
            s += Rr[j]     * (H[j] * T[j] + H[d + j] * T[d + j])
               + Rr[d + j] * (H[d + j] * T[j] - H[j] * T[d + j]);
        }
        return s;
    }

    void compose_path(const std::vector<int>& rels,
                      std::vector<double>& cre,
                      std::vector<double>& cim) const {
        const int d = dim_, d2 = d2_size();
        cre.assign(d, 1.0); cim.assign(d, 0.0);
        for (int r : rels) {
            const double* rr = &R_[(size_t)r * d2];
            std::vector<double> nre(d), nim(d);
            for (int j = 0; j < d; ++j) {
                nre[j] = cre[j] * rr[j] - cim[j] * rr[d + j];
                nim[j] = cre[j] * rr[d + j] + cim[j] * rr[j];
            }
            cre.swap(nre); cim.swap(nim);
        }
    }

    // Sampled-softmax ranking over {dst} ∪ negs; backprop through the
    // complex composition into E_dst/E_negs and every relation on the path.
    double rank_step(int src, const std::vector<int>& rels, int dst,
                     const std::vector<int>& negs, double tau, int RnA) {
        const int d = dim_, d2 = d2_size();
        // per-hop complex composition snapshots (needed for backprop sweep)
        std::vector<std::vector<double>> cre(rels.size() + 1,
                                             std::vector<double>(d, 1.0));
        std::vector<std::vector<double>> cim(rels.size() + 1,
                                             std::vector<double>(d, 0.0));
        for (size_t k = 0; k < rels.size(); ++k) {
            const double* rr = &R_[(size_t)rels[k] * d2];
            for (int j = 0; j < d; ++j) {
                cre[k + 1][j] = cre[k][j] * rr[j] - cim[k][j] * rr[d + j];
                cim[k + 1][j] = cre[k][j] * rr[d + j] + cim[k][j] * rr[j];
            }
        }

        auto tail_score = [&](int e) {
            const double* T = &E_[(size_t)e * d2];
            const double* H = &E_[(size_t)src * d2];
            double s = 0.;
            const auto& CRE = cre.back();
            const auto& CIM = cim.back();
            for (int j = 0; j < d; ++j) {
                s += CRE[j] * (H[j] * T[j] + H[d + j] * T[d + j])
                   + CIM[j] * (H[d + j] * T[j] - H[j] * T[d + j]);
            }
            return s / tau;
        };

        std::vector<int> cands; cands.push_back(dst);
        for (int n : negs) cands.push_back(n);
        const int C = (int)cands.size();
        std::vector<double> p(C);
        for (int k = 0; k < C; ++k) p[k] = tail_score(cands[k]);
        double mx = *std::max_element(p.begin(), p.end());
        double Z = 0.;
        for (auto& v : p) { v = std::exp(v - mx); Z += v; }
        for (auto& v : p) v /= Z;
        const double loss = -std::log(p[0] + 1e-12);

        auto backprop = [&](int e, double g) {
            // g is dLoss/d(s/tau); chain to raw score via 1/tau
            const double gg = g / tau;
            double* T = &E_[(size_t)e * d2];
            const double* H = &E_[(size_t)src * d2];
            std::vector<double> gre(d), gim(d);
            const auto& CRE = cre.back();
            const auto& CIM = cim.back();
            for (int j = 0; j < d; ++j) {
                // accumulate entity grads directly (no in-place mutation)
                gradE_[(size_t)e * d2 + j] +=
                    gg * (CRE[j] * H[j] + CIM[j] * H[d + j]);
                gradE_[(size_t)e * d2 + d + j] +=
                    gg * (CRE[j] * H[d + j] - CIM[j] * H[j]);
                // carry for relation sweep
                gre[j] = gg * (CRE[j] * T[j] + CIM[j] * T[d + j]);
                gim[j] = gg * (-CIM[j] * T[j] + CRE[j] * T[d + j]);
            }
            // relation sweep backwards through complex products:
            // forward was c_{k+1} = c_k ⊗ r_k
            // ⇒ ∂/∂r_k = gc_{k+1} ⊗ conj(c_k),  gc_k = gc_{k+1} ⊗ conj(r_k)
            for (int kk = (int)rels.size() - 1; kk >= 0; --kk) {
                double* gR = &gradR_[(size_t)rels[kk] * d2];
                const double* Rk = &R_[(size_t)rels[kk] * d2];
                for (int j = 0; j < d; ++j) {
                    const double ar = gre[j], ai = gim[j];
                    gR[j]      += ar * cre[kk][j] + ai * cim[kk][j];
                    gR[d + j]  += ai * cre[kk][j] - ar * cim[kk][j];
                    const double nr = ar * Rk[j] + ai * Rk[d + j];
                    const double ni = ai * Rk[j] - ar * Rk[d + j];
                    gre[j] = nr; gim[j] = ni;
                }
            }
        };
        backprop(dst, p[0] - 1.0);
        for (int k = 0; k < (int)negs.size(); ++k)
            backprop(negs[k], p[k + 1]);
        return loss;
    }
    void clip_grads(double max_norm = 5.0) {
        double sq = 0.;
        for (double v : gradE_) sq += v * v;
        for (double v : gradR_) sq += v * v;
        const double n = std::sqrt(sq);
        if (n > max_norm && n > 0) {
            const double s = max_norm / n;
            for (auto& v : gradE_) v *= s;
            for (auto& v : gradR_) v *= s;
        }
    }

    void norm_all(int V, int RnA, int /*d2*/) {
        auto norm_row = [&](std::vector<double>& Mv, size_t base) {
            double sq = 0.;
            for (size_t j = base; j < base + d2_size(); ++j) sq += Mv[j] * Mv[j];
            double nrm = std::sqrt(sq);
            if (nrm > 1e-12)
                for (size_t j = base; j < base + d2_size(); ++j) Mv[j] /= nrm;
        };
        for (int v = 0; v < V; ++v) norm_row(E_, (size_t)v * d2_size());
        for (int r = 0; r < RnA; ++r) norm_row(R_, (size_t)r * d2_size());
    }

    void adam_step(double lr) {
        ++adam_t_;
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        const double bc1 = 1.0 - std::pow(b1, adam_t_);
        const double bc2 = 1.0 - std::pow(b2, adam_t_);
        auto upd = [&](std::vector<double>& P, std::vector<double>& G,
                       std::vector<double>& M, std::vector<double>& Vv) {
            for (size_t i = 0; i < P.size(); ++i) {
                M[i] = b1 * M[i] + (1 - b1) * G[i];
                Vv[i] = b2 * Vv[i] + (1 - b2) * G[i] * G[i];
                P[i] -= lr * (M[i] / bc1) / (std::sqrt(Vv[i] / bc2) + eps);
            }
        };
        upd(E_, gradE_, mE_, vE_);
        upd(R_, gradR_, mR_, vR_);
    }

    std::vector<std::string> entities_, relations_;
    std::unordered_map<std::string, int> eid_, rid_;
    std::unordered_set<uint64_t> true_keys_;
    std::vector<Edge> edges_;
    std::vector<std::vector<std::pair<int,int>>> adj_out_;

    std::vector<double> E_, R_;
    std::vector<double> gradE_, gradR_, mE_, vE_, mR_, vR_;
    long long adam_t_ = 0;
    std::mt19937 rng_{7};
};

} // namespace reasoning
} // namespace engines
} // namespace brain3
