#pragma once
/*
 * episodic.hpp — Episodic Memory (Hippocampus), Component 2 of Brain v2
 *
 * Stores sequences of SOM activation vectors as episodes.
 * Retrieves the most similar past episode given current context.
 *
 * Storage policy: store when prediction_error > surprise_threshold.
 * This means only novel/unexpected events are remembered — matching
 * how the hippocampus tags events for consolidation.
 *
 * Retrieval: cosine similarity between current vector and episode start vectors.
 * Returns the best matching episode (sequence of vectors).
 *
 * Consolidation: during rest(), older episodes are summarized into
 * compressed prototypes (centroid of similar episodes). Reduces memory
 * use while preserving semantic content — semantic memory from episodic.
 */

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <string>

namespace brain2 {

struct Episode {
    std::vector<std::vector<float>> frames;  // sequence of SOM vectors
    float surprise;                          // prediction error at storage time
    int   timestamp;                         // step when stored

    float cosine_sim(const std::vector<float>& query) const {
        if (frames.empty()) return 0.f;
        const auto& first = frames[0];
        float dot = 0.f, na = 0.f, nb = 0.f;
        for (size_t i = 0; i < first.size() && i < query.size(); i++) {
            dot += first[i] * query[i];
            na  += first[i] * first[i];
            nb  += query[i] * query[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }
};

class EpisodicMemory {
public:
    int   n_dims;
    int   max_episodes;
    float surprise_threshold;

private:
    std::deque<Episode>              episodes_;
    std::vector<Episode>             prototypes_;  // consolidated semantic memory
    std::vector<std::vector<float>>  current_ep_;  // building current episode
    int                              step_;
    std::unique_ptr<std::mutex>      mtx_;

    // Cosine similarity between two vectors
    static float cosine(const std::vector<float>& a,
                        const std::vector<float>& b) noexcept {
        float dot = 0.f, na = 0.f, nb = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    // Centroid of a set of vectors
    static std::vector<float> centroid(const std::vector<std::vector<float>>& vecs) {
        if (vecs.empty()) return {};
        std::vector<float> c(vecs[0].size(), 0.f);
        for (const auto& v : vecs)
            for (size_t i = 0; i < c.size() && i < v.size(); i++)
                c[i] += v[i];
        float scale = 1.f / vecs.size();
        for (auto& x : c) x *= scale;
        return c;
    }

public:
    EpisodicMemory() : n_dims(0), max_episodes(0), surprise_threshold(0),
                       step_(0), mtx_(std::make_unique<std::mutex>()) {}

    EpisodicMemory(int n_dims, int max_episodes = 2000,
                   float surprise_threshold = 0.3f)
        : n_dims(n_dims), max_episodes(max_episodes),
          surprise_threshold(surprise_threshold),
          step_(0), mtx_(std::make_unique<std::mutex>()) {}

    EpisodicMemory(EpisodicMemory&&)            = default;
    EpisodicMemory& operator=(EpisodicMemory&&) = default;
    EpisodicMemory(const EpisodicMemory&)       = delete;
    EpisodicMemory& operator=(const EpisodicMemory&) = delete;

    // Add a frame to current building episode
    void observe(const std::vector<float>& activation) {
        std::lock_guard<std::mutex> lock(*mtx_);
        current_ep_.push_back(activation);
        if ((int)current_ep_.size() > 20)  // max episode length
            current_ep_.erase(current_ep_.begin());
        step_++;
    }

    // Commit current episode to memory if surprise is high enough
    // Returns true if episode was stored
    bool commit(float prediction_error) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (current_ep_.size() < 2) return false;
        if (prediction_error < surprise_threshold) {
            current_ep_.clear();
            return false;
        }

        Episode ep;
        ep.frames    = current_ep_;
        ep.surprise  = prediction_error;
        ep.timestamp = step_;
        episodes_.push_back(std::move(ep));
        current_ep_.clear();

        // Evict oldest if over capacity
        while ((int)episodes_.size() > max_episodes)
            episodes_.pop_front();

        return true;
    }

    // Retrieve most similar past episode to current query vector
    // Returns empty if no episodes stored
    const Episode* retrieve(const std::vector<float>& query) const {
        if (episodes_.empty()) return nullptr;
        const Episode* best = nullptr;
        float best_sim = -1.f;
        for (const auto& ep : episodes_) {
            float sim = ep.cosine_sim(query);
            if (sim > best_sim) { best_sim = sim; best = &ep; }
        }
        // Also check prototypes
        for (const auto& ep : prototypes_) {
            float sim = ep.cosine_sim(query);
            if (sim > best_sim) { best_sim = sim; best = &ep; }
        }
        return best;
    }

    // Retrieve top-k most similar episodes (sorted by similarity)
    std::vector<std::pair<float, int>> retrieve_topk(
            const std::vector<float>& query, int k = 3) const {
        std::vector<std::pair<float, int>> sims;
        sims.reserve(episodes_.size());
        for (int i = 0; i < (int)episodes_.size(); i++)
            sims.push_back({episodes_[i].cosine_sim(query), i});
        std::partial_sort(sims.begin(),
                          sims.begin() + std::min(k, (int)sims.size()),
                          sims.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        if ((int)sims.size() > k) sims.resize(k);
        return sims;
    }

    // Rest/consolidation: cluster similar episodes into prototypes
    // Call this during sleep/downtime — not in hot path
    int consolidate(float similarity_threshold = 0.85f) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if ((int)episodes_.size() < 10) return 0;

        std::vector<bool> merged(episodes_.size(), false);
        int count = 0;

        for (int i = 0; i < (int)episodes_.size(); i++) {
            if (merged[i]) continue;
            std::vector<std::vector<float>> cluster;
            cluster.push_back(episodes_[i].frames[0]);
            merged[i] = true;

            for (int j = i+1; j < (int)episodes_.size(); j++) {
                if (merged[j]) continue;
                if (cosine(episodes_[i].frames[0],
                           episodes_[j].frames[0]) > similarity_threshold) {
                    cluster.push_back(episodes_[j].frames[0]);
                    merged[j] = true;
                }
            }

            if ((int)cluster.size() >= 3) {
                Episode proto;
                proto.frames.push_back(centroid(cluster));
                proto.surprise  = 0.5f;
                proto.timestamp = step_;
                prototypes_.push_back(std::move(proto));
                count++;
            }
        }

        // Remove merged episodes (keep unmerged)
        std::deque<Episode> remaining;
        for (int i = 0; i < (int)episodes_.size(); i++)
            if (!merged[i]) remaining.push_back(std::move(episodes_[i]));
        episodes_ = std::move(remaining);

        return count;
    }

    int episode_count()   const noexcept { return (int)episodes_.size(); }
    int prototype_count() const noexcept { return (int)prototypes_.size(); }
    int step()            const noexcept { return step_; }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("EpisodicMemory::save: cannot open " + path);
        f.write((const char*)&n_dims,              sizeof(int));
        f.write((const char*)&max_episodes,        sizeof(int));
        f.write((const char*)&surprise_threshold,  sizeof(float));
        f.write((const char*)&step_,               sizeof(int));

        auto write_episodes = [&](const auto& eps) {
            int n = (int)eps.size();
            f.write((const char*)&n, sizeof(int));
            for (const auto& ep : eps) {
                int nf = (int)ep.frames.size();
                f.write((const char*)&nf, sizeof(int));
                for (const auto& fr : ep.frames) {
                    int fd = (int)fr.size();
                    f.write((const char*)&fd, sizeof(int));
                    f.write((const char*)fr.data(), (std::streamsize)(fd * sizeof(float)));
                }
                f.write((const char*)&ep.surprise,  sizeof(float));
                f.write((const char*)&ep.timestamp, sizeof(int));
            }
        };
        write_episodes(episodes_);
        write_episodes(prototypes_);
    }

    static EpisodicMemory load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("EpisodicMemory::load: cannot open " + path);
        EpisodicMemory m;
        f.read((char*)&m.n_dims,             sizeof(int));
        f.read((char*)&m.max_episodes,       sizeof(int));
        f.read((char*)&m.surprise_threshold, sizeof(float));
        f.read((char*)&m.step_,              sizeof(int));
        m.mtx_ = std::make_unique<std::mutex>();

        auto read_episodes = [&](auto& eps) {
            int n; f.read((char*)&n, sizeof(int));
            for (int i = 0; i < n; i++) {
                Episode ep;
                int nf; f.read((char*)&nf, sizeof(int));
                ep.frames.resize(nf);
                for (auto& fr : ep.frames) {
                    int fd; f.read((char*)&fd, sizeof(int));
                    fr.resize(fd);
                    f.read((char*)fr.data(), (std::streamsize)(fd * sizeof(float)));
                }
                f.read((char*)&ep.surprise,  sizeof(float));
                f.read((char*)&ep.timestamp, sizeof(int));
                eps.push_back(std::move(ep));
            }
        };
        read_episodes(m.episodes_);
        read_episodes(m.prototypes_);
        return m;
    }
};

} // namespace brain2
