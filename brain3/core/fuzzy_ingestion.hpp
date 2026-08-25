#pragma once
/**
 * brain3/core/fuzzy_ingestion.hpp
 *
 * FUZZY INGESTION PIPELINE — Stage 2 between crisp extraction and sleep.
 *
 * Activates the Tier-B organs during data diet:
 *   - SOM concept map: every entity lands somewhere; bmu_distance IS the
 *     novelty signal (quantization error — far from everything known)
 *   - Entity resolution: same-BMU / near-BMU entities become merge
 *     candidates (attacks the classic KG fragmentation killer)
 *   - Episodic commits: batches become replayable episodes, feeding the
 *     existing sleep consolidation
 *
 * Honest scope: the LSTM Predictor stays out of v1 — its step() is bound
 * to the LM vocabulary, not concept streams. SOM quantization error is the
 * organ-native novelty signal instead.
 */
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <deque>
#include <vector>

#include "../fuzzy/core/som.hpp"

namespace brain3 {
namespace core {

class FuzzyIngestionPipeline {
public:
    struct Verdict {
        bool novel_entity = false;      // first time seeing this surface
        float bmu_dist = 0.f;           // SOM quantization error (novelty)
        int bmu = -1;
        std::string merge_hint;         // existing entity this may duplicate
    };

    explicit FuzzyIngestionPipeline(int som_rows = 12, int som_cols = 12,
                                    int embed_dim = 32)
        : som_(som_rows, som_cols, embed_dim),
          dim_(embed_dim) {
        // warm the map with spread synthetic anchors so early entities
        // don't collapse onto neuron zero
        std::mt19937 g(7);
        for (int k = 0; k < 60; ++k) {
            auto v = random_vec(g);
            som_.update(v, som_.find_bmu(v));
        }
    }

    // ── entity observation: embedding + SOM placement + resolution hint ────
    Verdict observe_entity(const std::string& name) {
        Verdict v;
        auto it = ent_vec_.find(name);
        if (it == ent_vec_.end()) {
            auto vec = char_embed(name);
            ent_vec_[name] = vec;
            v.novel_entity = true;
            it = ent_vec_.find(name);
        }
        const auto& vec = it->second;

        v.bmu = som_.find_bmu(vec);
        v.bmu_dist = som_.bmu_distance(vec);
        // register into this cell AND its 4-neighbors so near-strings that
        // land adjacently still co-locate for resolution
        for (int cell : neighborhood(v.bmu)) bmu_entities_[cell].push_back(name);

        // resolution hint: nearest already-known neighbor in same bucket
        auto& bucket = bmu_entities_[v.bmu];
        for (auto rit = bucket.rbegin(); rit != bucket.rend(); ++rit) {
            const std::string& other = *rit;
            if (other == name || !ent_vec_.count(other)) continue;
            v.merge_hint = other;
            break;                              // most recent prior resident
        }
        som_.update(vec, v.bmu);                // adapt map toward input

        last_dist_ = v.bmu_dist;
        dist_history_.push_back(v.bmu_dist);
        if (dist_history_.size() > 512) dist_history_.pop_front();
        return v;
    }

    // triple-level novelty: mean BMU distance of head/tail + relation rarity
    double observe_triple(const std::string& h, const std::string& r,
                          const std::string& t) {
        auto vh = ensure(h), vt = ensure(t);
        auto vhv = ent_vec_.at(vh), vtv = ent_vec_.at(vt);
        float d = 0.5f * (som_.bmu_distance(vhv) + som_.bmu_distance(vtv));
        (void)vhv; (void)vtv;

        // relation rarity: P(r | h-class) proxy via global counts
        rel_count_[r]++;
        total_rels_++;
        double rarity = 1.0 - (double)rel_count_[r] / (double)total_rels_;
        rarity = std::max(0.0, rarity);

        const double novelty = 0.5 * (d / std::max(1.f, som_scale_)) + 0.5 * rarity;
        batch_errors_.push_back(novelty);
        last_dist_ = (float)novelty;
        return novelty;
    }

    // ── entity-resolution candidates ────────────────────────────────────────
    // groups of entities sharing a BMU whose pairwise char-embedding
    // distance is below threshold (near-strings on the same concept slot)
    std::vector<int> neighborhood(int bmu) const {
        int r = bmu / std::max(1, som_.cols), c = bmu % std::max(1, som_.cols);
        std::vector<int> cells{bmu};
        int dr[] = {-1,1,0,0}, dc[] = {0,0,-1,1};
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr >= 0 && nr < som_.rows && nc >= 0 && nc < som_.cols)
                cells.push_back(nr * som_.cols + nc);
        }
        return cells;
    }

    std::vector<std::vector<std::string>>
    resolution_candidates(float max_pair_dist = 0.35f) const {
        std::vector<std::vector<std::string>> out;
        for (const auto& [bmu, names] : bmu_entities_) {
            if (names.size() < 2) continue;
            std::vector<std::string> group;
            std::set<std::string> uniq(names.begin(), names.end());
            std::vector<std::string> list(uniq.begin(), uniq.end());
            for (size_t i = 0; i < list.size(); ++i)
                for (size_t j = i + 1; j < list.size(); ++j) {
                    if (embed_dist(list[i], list[j]) <= max_pair_dist) {
                        if (group.empty()) { group.push_back(list[i]); group.push_back(list[j]); }
                        else {
                            bool has_i = false, has_j = false;
                            for (auto& w : group) { if (w==list[i]) has_i=true; if (w==list[j]) has_j=true; }
                            if (!has_i) group.push_back(list[i]);
                            if (!has_j) group.push_back(list[j]);
                        }
                    }
                }
            if (group.size() >= 2) out.push_back(group);
        }
        return out;
    }

    // text-level novelty: BMU distance of the line's bigram embedding
    // against the concept map built from REAL ingestion
    double text_novelty(const std::string& line) const {
        return som_.bmu_distance(char_embed(line));
    }

    // batch boundary: mean novelty since last commit (episodic error signal)
    double flush_batch() {
        double m = 0.;
        if (!batch_errors_.empty()) {
            for (double e : batch_errors_) m += e;
            m /= (double)batch_errors_.size();
        }
        batch_errors_.clear();
        return m;
    }
    double recent_mean_dist() const {
        if (dist_history_.empty()) return 0.;
        double s = 0.; for (float f : dist_history_) s += f;
        return s / dist_history_.size();
    }

private:
    static constexpr float som_scale_ = 2.8f;   // empirical max bmu_dist scale

    std::vector<float> random_vec(std::mt19937& g) {
        std::normal_distribution<float> nd(0.f, 0.5f);
        std::vector<float> v(dim_);
        for (auto& x : v) x = nd(g);
        return v;
    }
    std::vector<float> last_probe_;

    // deterministic CHARACTER-BIGRAM embedding: near-strings share most
    // bigrams ⇒ cosine-close vectors ⇒ same SOM neighborhood. This is what
    // makes entity-resolution work for subset/short-form names.
    std::vector<float> char_embed(const std::string& word) const {
        std::vector<float> v(dim_, 0.f);
        auto bump = [&](const std::string& g) {
            size_t h = 1469598103934665603ULL;
            for (char c : g) { h ^= (unsigned char)c; h *= 1099511628211ULL; }
            v[h % dim_] += 1.f;
        };
        std::string s = "_" + word + "_";
        for (size_t i = 0; i + 1 < s.size(); ++i) bump(s.substr(i, 2));
        float n = 0.f;
        for (auto& x : v) n += x * x;
        n = std::sqrt(n);
        if (n > 1e-9f) for (auto& x : v) x /= n;
        return v;
    }

    // cosine distance between two entities' bigram embeddings
    double embed_dist(const std::string& a, const std::string& b) const {
        auto va = char_embed(a), vb = char_embed(b);
        double dot = 0., na = 0., nb = 0.;
        for (int k = 0; k < dim_; ++k) {
            dot += va[k] * vb[k];
            na += va[k] * va[k];
            nb += vb[k] * vb[k];
        }
        if (na < 1e-12 || nb < 1e-12) return 1.0;
        return 1.0 - dot / (std::sqrt(na) * std::sqrt(nb));
    }

    std::string ensure(const std::string& name) {
        if (!ent_vec_.count(name)) observe_entity(name);
        return name;
    }

    brain2::SOM som_;
    int dim_;
    std::unordered_map<std::string, std::vector<float>> ent_vec_;
    std::unordered_map<int, std::vector<std::string>> bmu_entities_;
    std::map<std::string, long long> rel_count_;
    long long total_rels_ = 0;
    std::vector<double> batch_errors_;
    std::deque<float> dist_history_;
    float last_dist_ = 0.f;
};

} // namespace core
} // namespace brain3
