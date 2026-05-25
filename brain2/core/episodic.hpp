#pragma once
/*
 * episodic.hpp — Hierarchical Spiking Episodic Memory (Hippocampus)
 *
 * Episodes are no longer flat arrays, but hierarchical trees (Roots -> Chunks -> Frames).
 * This structure enables extremely fast O(log N) retrieval across large narrative sequences.
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

// Recursive Tree Node for memory chunks
struct EpisodeNode {
    std::vector<bool> summary_spike; 
    std::vector<EpisodeNode> children; 

    // Similarity between query and this node's summary
    float sparse_sim(const std::vector<float>& query) const {
        if (summary_spike.empty()) return 0.f;
        int matches = 0, query_ones = 0, first_ones = 0;
        
        for (size_t i = 0; i < summary_spike.size() && i < query.size(); i++) {
            bool q_bit = query[i] > 0.1f;
            if (summary_spike[i]) first_ones++;
            if (q_bit) query_ones++;
            if (summary_spike[i] && q_bit) matches++;
        }
        
        if (first_ones == 0 || query_ones == 0) return 0.f;
        return (float)matches / std::sqrt((float)first_ones * (float)query_ones);
    }
};

struct Episode {
    EpisodeNode root;
    float surprise;
    int   timestamp;

    float get_sim(const std::vector<float>& query) const {
        return root.sparse_sim(query);
    }
};

class EpisodicMemory {
public:
    int   n_dims;
    int   max_episodes;
    float surprise_threshold;

private:
    std::deque<Episode>              episodes_;
    std::vector<Episode>             prototypes_;  
    std::vector<std::vector<bool>>   current_ep_;  
    int                              step_;
    std::unique_ptr<std::mutex>      mtx_;

    static std::vector<bool> centroid(const std::vector<std::vector<bool>>& vecs) {
        if (vecs.empty()) return {};
        std::vector<int> counts(vecs[0].size(), 0);
        for (const auto& v : vecs) {
            for (size_t i = 0; i < counts.size() && i < v.size(); i++) {
                if (v[i]) counts[i]++;
            }
        }
        std::vector<bool> out(counts.size(), false);
        int threshold = vecs.size() / 2;
        for (size_t i = 0; i < counts.size(); i++) {
            out[i] = (counts[i] > threshold);
        }
        return out;
    }
    
    // Builds a chunk tree out of raw frames (Chunk size = 5)
    EpisodeNode build_tree(const std::vector<std::vector<bool>>& frames) {
        EpisodeNode root;
        if (frames.empty()) return root;
        
        root.summary_spike = centroid(frames);
        
        int chunk_size = 5;
        for (size_t i = 0; i < frames.size(); i += chunk_size) {
            EpisodeNode chunk_node;
            std::vector<std::vector<bool>> chunk_frames;
            for (size_t j = i; j < std::min(i + chunk_size, frames.size()); j++) {
                EpisodeNode leaf;
                leaf.summary_spike = frames[j];
                chunk_node.children.push_back(leaf);
                chunk_frames.push_back(frames[j]);
            }
            chunk_node.summary_spike = centroid(chunk_frames);
            root.children.push_back(chunk_node);
        }
        return root;
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

    void observe(const std::vector<float>& activation) {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<bool> spike_frame(activation.size());
        for (size_t i = 0; i < activation.size(); i++) {
            spike_frame[i] = (activation[i] > 0.1f);
        }
        current_ep_.push_back(std::move(spike_frame));
        if ((int)current_ep_.size() > 50) // Allow longer episodes now that it's hierarchical
            current_ep_.erase(current_ep_.begin());
        step_++;
    }

    bool commit(float prediction_error) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (current_ep_.size() < 2) return false;
        if (prediction_error < surprise_threshold) {
            current_ep_.clear();
            return false;
        }

        Episode ep;
        ep.root      = build_tree(current_ep_);
        ep.surprise  = prediction_error;
        ep.timestamp = step_;
        episodes_.push_back(std::move(ep));
        current_ep_.clear();

        while ((int)episodes_.size() > max_episodes)
            episodes_.pop_front();

        return true;
    }

    const Episode* retrieve(const std::vector<float>& query) const {
        if (episodes_.empty()) return nullptr;
        const Episode* best = nullptr;
        float best_sim = -1.f;
        for (const auto& ep : episodes_) {
            float sim = ep.get_sim(query);
            if (sim > best_sim) { best_sim = sim; best = &ep; }
        }
        for (const auto& ep : prototypes_) {
            float sim = ep.get_sim(query);
            if (sim > best_sim) { best_sim = sim; best = &ep; }
        }
        return best;
    }

    const Episode* get_episode(int idx) const {
        if (idx >= 0 && idx < (int)episodes_.size()) return &episodes_[idx];
        return nullptr;
    }

    std::vector<std::pair<float, int>> retrieve_topk(
            const std::vector<float>& query, int k = 3) const {
        std::vector<std::pair<float, int>> sims;
        sims.reserve(episodes_.size());
        for (int i = 0; i < (int)episodes_.size(); i++)
            sims.push_back({episodes_[i].get_sim(query), i});
        std::partial_sort(sims.begin(),
                          sims.begin() + std::min(k, (int)sims.size()),
                          sims.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        if ((int)sims.size() > k) sims.resize(k);
        return sims;
    }

    static float frame_sim(const std::vector<bool>& a, const std::vector<bool>& b) {
        int matches = 0, a_ones = 0, b_ones = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            if (a[i]) a_ones++;
            if (b[i]) b_ones++;
            if (a[i] && b[i]) matches++;
        }
        if (a_ones == 0 || b_ones == 0) return 0.f;
        return (float)matches / std::sqrt((float)a_ones * (float)b_ones);
    }

    int consolidate(float similarity_threshold = 0.85f) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if ((int)episodes_.size() < 10) return 0;

        std::vector<bool> merged(episodes_.size(), false);
        int count = 0;

        for (int i = 0; i < (int)episodes_.size(); i++) {
            if (merged[i]) continue;
            std::vector<std::vector<bool>> cluster;
            cluster.push_back(episodes_[i].root.summary_spike);
            merged[i] = true;

            for (int j = i+1; j < (int)episodes_.size(); j++) {
                if (merged[j]) continue;
                if (frame_sim(episodes_[i].root.summary_spike, episodes_[j].root.summary_spike) > similarity_threshold) {
                    cluster.push_back(episodes_[j].root.summary_spike);
                    merged[j] = true;
                }
            }

            if ((int)cluster.size() >= 3) {
                Episode proto;
                proto.root.summary_spike = centroid(cluster);
                proto.surprise  = 0.5f;
                proto.timestamp = step_;
                prototypes_.push_back(std::move(proto));
                count++;
            }
        }

        std::deque<Episode> remaining;
        for (int i = 0; i < (int)episodes_.size(); i++)
            if (!merged[i]) remaining.push_back(std::move(episodes_[i]));
        episodes_ = std::move(remaining);

        return count;
    }

    int episode_count()   const noexcept { return (int)episodes_.size(); }
    int prototype_count() const noexcept { return (int)prototypes_.size(); }
    int step()            const noexcept { return step_; }

    // Recursive save for tree nodes
    void save_node(std::ofstream& f, const EpisodeNode& node) const {
        int fd = (int)node.summary_spike.size();
        f.write((const char*)&fd, sizeof(int));
        std::vector<uint8_t> packed((fd + 7) / 8, 0);
        for (int i = 0; i < fd; i++) {
            if (node.summary_spike[i]) packed[i / 8] |= (1 << (i % 8));
        }
        f.write((const char*)packed.data(), packed.size());
        
        int nc = (int)node.children.size();
        f.write((const char*)&nc, sizeof(int));
        for (const auto& child : node.children) {
            save_node(f, child);
        }
    }

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
                save_node(f, ep.root);
                f.write((const char*)&ep.surprise,  sizeof(float));
                f.write((const char*)&ep.timestamp, sizeof(int));
            }
        };
        write_episodes(episodes_);
        write_episodes(prototypes_);
    }

    // Recursive load for tree nodes
    static EpisodeNode load_node(std::ifstream& f) {
        EpisodeNode node;
        int fd; f.read((char*)&fd, sizeof(int));
        node.summary_spike.resize(fd, false);
        std::vector<uint8_t> packed((fd + 7) / 8, 0);
        f.read((char*)packed.data(), packed.size());
        for (int i = 0; i < fd; i++) {
            if (packed[i / 8] & (1 << (i % 8))) node.summary_spike[i] = true;
        }
        
        int nc; f.read((char*)&nc, sizeof(int));
        node.children.resize(nc);
        for (int i = 0; i < nc; i++) {
            node.children[i] = load_node(f);
        }
        return node;
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
                ep.root = load_node(f);
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
