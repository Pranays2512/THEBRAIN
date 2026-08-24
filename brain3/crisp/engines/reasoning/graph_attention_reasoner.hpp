#pragma once
/**
 * brain3/crisp/engines/reasoning/graph_attention_reasoner.hpp
 *
 * GRAPH-ATTENTION REASONER — multi-hop inference over the knowledge base.
 *
 * The organ that turns "stores facts" into "connects facts":
 *   - relation embeddings compose elementwise along a path (DistMult), so
 *     ANY relation sequence is scoreable — including pairs never co-trained
 *     (structural generalization by construction)
 *   - inference propagates an attention frontier: each edge competes via
 *     softmax among its siblings (learned spreading activation); per-hop
 *     attention snapshots are exposed for verification
 *   - training is self-supervised: random walks + corrupted negatives,
 *     chain-rule gradients through the products (no parameter division),
 *     global-norm clipping
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

namespace brain3 {
namespace engines {
namespace reasoning {

class GraphAttentionReasoner {
public:
    // ── graph store ────────────────────────────────────────────────────────
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
        edges_.push_back({head, rel, tail});
        // adjacency stores (RELATION, tail) — consumers (query filter,
        // walk sampler) rely on this contract
        adj_out_[head].push_back({rel, tail});
    }
    int entity_count() const { return (int)entities_.size(); }
    int edge_count() const { return (int)edges_.size(); }

    void load_from_facts(
        const std::set<std::tuple<std::string, std::string, std::string>>& facts) {
        for (const auto& [s, r, o] : facts)
            add_edge(add_entity(s), add_relation(r), add_entity(o));
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
        const int V = entity_count(), Rn = (int)relations_.size(), d = dim_;
        // AUGMENTED graph: add an inverse relation per relation so head-role
        // and tail-role geometry separate (fixes sibling-tops-head artifact
        // of symmetric DistMult on hierarchies). Forward ids stay < Rn;
        // inverse ids live in [Rn, 2Rn) and are never queried directly.
        const int RnA = 2 * Rn;
        std::vector<std::vector<std::pair<int,int>>> out(V);   // head -> (rel,tail)
        std::unordered_set<uint64_t> true_keys;                // rebuilt incl. inverses
        auto key = [this](int h, int r, int t) {
            return (uint64_t)h * 1000003ULL * 1000003ULL +
                   (uint64_t)r * 1000003ULL + (uint64_t)t;
        };
        for (const auto& e : edges_) {
            out[e.head].push_back({e.rel, e.tail});
            out[e.tail].push_back({Rn + e.rel, e.head});       // inverse edge
            true_keys.insert(key(e.head, e.rel, e.tail));
            true_keys.insert(key(e.tail, Rn + e.rel, e.head));
        }

        std::normal_distribution<double> nd(0.0, std::sqrt(1.0 / d));
        E_.resize((size_t)V * d);
        for (auto& v : E_) v = nd(rng_);
        R_.resize((size_t)RnA * d);
        for (auto& v : R_) v = nd(rng_);
        mE_.assign(E_.size(), 0.0); vE_.assign(E_.size(), 0.0);
        mR_.assign(R_.size(), 0.0); vR_.assign(R_.size(), 0.0);
        gradE_.assign(E_.size(), 0.0);
        gradR_.assign((size_t)RnA * d, 0.0);
        adam_t_ = 0;

        const double tau = 0.35;
        for (int step = 1; step <= cfg.steps; ++step) {
            std::fill(gradE_.begin(), gradE_.end(), 0.0);
            std::fill(gradR_.begin(), gradR_.end(),
                      0.0);
            double loss_sum = 0.0;
            for (int b = 0; b < cfg.batch; ++b) {
                auto [path_rel, nodes] = sample_walk(out, cfg.max_path_len);
                if ((int)path_rel.size() < 1) continue;
                const int src = nodes.front(), dst = nodes.back();

                std::vector<int> negs;
                for (int k = 0; k < cfg.negatives_per_pos; ++k) {
                    int cand = (int)(rng_() % V);
                    if (cand != dst &&
                        !true_keys.count(key(src, path_rel.back(), cand)))
                        negs.push_back(cand);
                }
                if (negs.empty()) continue;
                loss_sum += rank_step(src, path_rel, dst, negs, tau);
            }
            clip_grads();
            adam_step(cfg.lr);

            // project embeddings/relations onto unit spheres: bounds DistMult
            // scores and keeps attention softmaxes informative
            {
                const int dd = dim_;
                auto norm_row = [&](std::vector<double>& Mv, size_t base) {
                    double sq = 0.;
                    for (int j = 0; j < dd; ++j) sq += Mv[base + j] * Mv[base + j];
                    double nrm = std::sqrt(sq);
                    if (nrm > 1e-12)
                        for (int j = 0; j < dd; ++j) Mv[base + j] /= nrm;
                };
                for (int v2 = 0; v2 < V; ++v2) norm_row(E_, (size_t)v2 * dd);
                for (int r2 = 0; r2 < (int)relations_.size() * 2; ++r2) norm_row(R_, (size_t)r2 * dd);
            }
            if (step % 200 == 0 || step == 1) {
                double e2 = 0., r2 = 0., g2 = 0.;
                for (double v : E_) e2 += v * v;
                for (double v : R_) r2 += v * v;
                for (double v : gradE_) g2 += v * v;
                std::cerr << "[gar] step " << step << " |E|=" << std::sqrt(e2)
                          << " |R|=" << std::sqrt(r2)
                          << " |gE|=" << std::sqrt(g2) << "\n";
            }
        }
    }

    // ── querying ───────────────────────────────────────────────────────────
    struct Hit { int entity; double mass; };
    struct QueryResult {
        std::vector<Hit> ranked;
        std::vector<std::map<int, double>> stages;   // attention after each hop
    };

    QueryResult query_stages(int src, const std::vector<int>& rels,
                             double temperature = 1.0) const {
        std::map<int, double> frontier{{src, 1.0}};
        QueryResult qr;
        qr.stages.push_back(frontier);

        for (int rel : rels) {
            std::map<int, double> raw;                 // tail -> activation
            for (const auto& [u, mass] : frontier) {
                const auto& out = adj_out_[u];
                if (out.empty()) continue;
                // sibling softmax across u's outgoing edges (attention)
                double mx = -1e30;
                std::vector<double> en(out.size());
                bool any = false;
                for (size_t i = 0; i < out.size(); ++i) {
                    if (rel >= 0 && out[i].first != rel) { en[i] = -1e30; continue; }
                    const Edge& e = edges_[out[i].first];
                    double sc = score_triple(u, e.rel, e.tail);
                    en[i] = sc; mx = std::max(mx, sc); any = true;
                }
                if (!any || !std::isfinite(mx)) continue;
                double Z = 0.;
                for (double& v2 : en) { v2 = std::exp((v2 - mx) / temperature); Z += v2; }
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

    // compositional path score: E_src · (⊙R_path) ⊙ E_dst
    double path_score(int src, const std::vector<int>& rels, int dst) const {
        const int d = dim_;
        std::vector<double> comp(d, 1.0);
        for (int r : rels)
            for (int j = 0; j < d; ++j) comp[j] *= R_[(size_t)r * d + j];
        double s = 0.;
        for (int j = 0; j < d; ++j)
            s += E_[(size_t)src * d + j] * comp[j] * E_[(size_t)dst * d + j];
        return s;
    }

    int entity_id(const std::string& n) const {
        auto it = eid_.find(n); return it == eid_.end() ? -1 : it->second;
    }
    int relation_id(const std::string& n) const {
        auto it = rid_.find(n); return it == rid_.end() ? -1 : it->second;
    }
    const std::string& entity_name(int id) const { return entities_[id]; }

private:
    static constexpr int dim_ = 128;

    std::pair<std::vector<int>, std::vector<int>>
    sample_walk(const std::vector<std::vector<std::pair<int,int>>>& out, int L) {
        if (out.empty()) return {};
        int cur = (int)(rng_() % out.size());
        std::vector<int> rels, nodes{cur};
        for (int hop = 0; hop < L; ++hop) {
            const auto& o = out[cur];
            if (o.empty()) break;
            auto [r, t] = o[rng_() % o.size()];
            if (t < 0 || t >= (int)entities_.size())
                std::cerr << "[BUG] tail=" << t << " V=" << entities_.size()
                          << " head=" << cur << "\n";
            rels.push_back(r); nodes.push_back(t); cur = t;
        }
        // mixed-length supervision: sometimes truncate to hop-1 so single
        // relations receive direct positive signal alongside compositions
        if (rels.size() >= 2 && rng_() % 100 < 40) {
            rels.resize(1);
            nodes.resize(2);
        }
        return {rels, nodes};
    }

    double score_triple(int h, int r, int t) const {
        const int d = dim_;
        const double* eh = &E_[(size_t)h * d];
        const double* er = &R_[(size_t)r * d];
        const double* et = &E_[(size_t)t * d];
        double s = 0.;
        for (int j = 0; j < d; ++j) s += eh[j] * er[j] * et[j];
        return s;
    }

    // Sampled-softmax RANKING step (RotatE-style): the true destination
    // competes against corrupted negatives; loss = -log p(dst).
    // Gradients vanish once dst outranks sampled negatives — no saturation
    // churn; rankings are exactly what get optimized.
    double rank_step(int src, const std::vector<int>& rels, int dst,
                     const std::vector<int>& negs, double tau) {
        const int d = dim_;
        std::vector<std::vector<double>> comps(rels.size() + 1,
                                               std::vector<double>(d, 1.0));
        for (size_t k = 0; k < rels.size(); ++k)
            for (int j = 0; j < d; ++j)
                comps[k + 1][j] = comps[k][j] * R_[(size_t)rels[k] * d + j];

        auto score = [&](int e) {
            double s = 0.;
            for (int j = 0; j < d; ++j)
                s += E_[(size_t)e * d + j] * comps.back()[j] * E_[(size_t)e * d + j];
            return s / tau;
        };

        std::vector<int> cands; cands.push_back(dst);
        for (int n : negs) cands.push_back(n);
        const int C = (int)cands.size();
        std::vector<double> sc(C);
        for (int k = 0; k < C; ++k) sc[k] = score(cands[k]);
        double mx = *std::max_element(sc.begin(), sc.end());
        double Z = 0.;
        for (auto& v : sc) { v = std::exp(v - mx); Z += v; }
        for (auto& v : sc) v /= Z;

        double loss = -std::log(sc[0] + 1e-12);

        auto backprop_candidate = [&](int e, double g) {
            for (int j = 0; j < d; ++j) {
                const double common = g * comps.back()[j];
                gradE_[(size_t)e * d + j] += common * E_[(size_t)e * d + j];
            }
            std::vector<double> gc(d);
            for (int j = 0; j < d; ++j)
                gc[j] = g * E_[(size_t)e * d + j] * E_[(size_t)e * d + j];
            for (int kk = (int)rels.size() - 1; kk >= 0; --kk) {
                double* gR = &gradR_[(size_t)rels[kk] * d];
                const double* Rk = &R_[(size_t)rels[kk] * d];
                for (int j = 0; j < d; ++j) {
                    gR[j] += gc[j] * comps[kk][j];
                    gc[j] *= Rk[j];
                }
            }
        };
        // dL/ds_pos = p_pos - 1 ; dL/ds_neg = p_neg
        backprop_candidate(dst, sc[0] - 1.0);
        for (int k = 0; k < (int)negs.size(); ++k)
            backprop_candidate(negs[k], sc[k + 1]);

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
        // mild weight decay keeps DistMult magnitudes bounded so attention
        // softmaxes stay informative instead of collapsing one-hot
        const double wd = 1e-4;
        for (size_t i = 0; i < E_.size(); ++i)
            E_[i] -= lr * wd * E_[i];
        for (size_t i = 0; i < R_.size(); ++i)
            R_[i] -= lr * wd * R_[i];
        upd(E_, gradE_, mE_, vE_);
        upd(R_, gradR_, mR_, vR_);
    }

    std::vector<std::string> entities_, relations_;
    std::unordered_map<std::string, int> eid_, rid_;
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
