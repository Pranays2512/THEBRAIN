#pragma once
/*
 * som.hpp — Navigable Small World (NSW) Graph SOM
 *
 * Implements a dynamic graph where neurons (nodes) are connected by lateral edges.
 * Lookup is incredibly fast O(log N) via greedy graph traversal.
 * Plasticity (learning) applies to a node and its direct topological graph neighbors,
 * completely eliminating the constraints of a rigid 2D grid or static tree branches!
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <memory>
#include "lsh.hpp"

namespace brain2 {

struct SomNode {
    std::vector<float> weights;
    std::vector<int>   neighbors; // Indices of connected nodes (lateral edges)
    float              hits;      // Usage frequency
    int                last_visited;
};

class SOM {
public:
    int rows, cols;
    int n_neurons, n_dims;
    int max_neighbors;

private:
    std::vector<SomNode>         nodes_;
    int                          entry_point_; // Starting node for searches
    float                        lr_, lr_decay_;
    int                          step_;
    std::unique_ptr<std::mutex>  update_mtx_;
    mutable CognitiveTLB         tlb_;

    inline float l2sq(const float* __restrict__ a,
                      const float* __restrict__ b) const noexcept {
        float s = 0.f;
        for (int i = 0; i < n_dims; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    inline float vec_l2sq(const std::vector<float>& a, const float* b) const noexcept {
        return l2sq(a.data(), b);
    }

public:
    SOM() : rows(0), cols(0), n_neurons(0), n_dims(0), max_neighbors(16), entry_point_(0),
            lr_(0), lr_decay_(0), step_(0),
            update_mtx_(std::make_unique<std::mutex>()), tlb_() {}

    SOM(int rows, int cols, int n_dims,
        float init_lr      = 0.15f,
        float lr_decay     = 0.9999f,
        float radius_decay = 0.9999f, // Unused in Graph SOM, kept for API compat
        unsigned seed      = 42)
        : rows(rows), cols(cols), n_neurons(rows * cols), n_dims(n_dims), max_neighbors(16),
          entry_point_(0),
          lr_(init_lr), lr_decay_(lr_decay), step_(0),
          update_mtx_(std::make_unique<std::mutex>()),
          tlb_(n_dims, 64, seed)
    {
        nodes_.resize(n_neurons);
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.f, 0.3f);
        std::uniform_int_distribution<int> rand_node(0, n_neurons - 1);

        for (int i = 0; i < n_neurons; i++) {
            nodes_[i].weights.resize(n_dims);
            for (int j = 0; j < n_dims; j++) {
                nodes_[i].weights[j] = dist(rng);
            }
            nodes_[i].hits = 0.f;
            nodes_[i].last_visited = 0;
            
            // Initialize with random lateral connections (Small World property)
            for (int k = 0; k < max_neighbors; k++) {
                int neighbor = rand_node(rng);
                if (neighbor != i) {
                    nodes_[i].neighbors.push_back(neighbor);
                }
            }
        }
    }

    SOM(SOM&&)            = default;
    SOM& operator=(SOM&&) = default;
    SOM(const SOM&)       = delete;
    SOM& operator=(const SOM&) = delete;

    // Fast O(log N) Greedy Graph Search
    int find_bmu(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::find_bmu: dim mismatch");
            
        const float* inp = input.data();
        
        // 1. Cognitive TLB Lookup
        uint64_t logical_address = tlb_.hash(inp);
        int cached_bmu = tlb_.lookup(logical_address);
        
        if (cached_bmu != -1 && cached_bmu < n_neurons) {
            float dist = vec_l2sq(nodes_[cached_bmu].weights, inp);
            if (dist < 0.2f) return cached_bmu; 
        }
        
        // 2. Greedy NSW Graph Traversal
        int curr_node = entry_point_;
        float curr_dist = vec_l2sq(nodes_[curr_node].weights, inp);
        
        while (true) {
            int best_neighbor = -1;
            float best_dist = curr_dist;
            
            for (int neighbor : nodes_[curr_node].neighbors) {
                float d = vec_l2sq(nodes_[neighbor].weights, inp);
                if (d < best_dist) {
                    best_dist = d;
                    best_neighbor = neighbor;
                }
            }
            
            if (best_neighbor == -1) {
                break; // Local minimum reached! This is the BMU.
            }
            curr_node = best_neighbor;
            curr_dist = best_dist;
        }
        
        tlb_.cache(logical_address, curr_node);
        return curr_node;
    }

    // Graph-based sparse activation
    std::vector<float> activation_map(const std::vector<float>& input) const {
        std::vector<float> acts(n_neurons, 0.f);
        int bmu = find_bmu(input);
        
        // Activate BMU
        acts[bmu] = 1.0f;
        
        // Activate neighbors (decaying outward)
        const float* inp = input.data();
        for (int neighbor : nodes_[bmu].neighbors) {
            float d = vec_l2sq(nodes_[neighbor].weights, inp);
            acts[neighbor] = std::exp(-d * 2.0f); // Fast decay
        }
        
        return acts;
    }

    inline float grid_dist(int local_i, int local_j) const noexcept {
        return 0.f; // API compat
    }

    // Update weights and dynamically rewire graph
    void update(const std::vector<float>& input, int bmu, float reward_mod = 1.f) {
        std::lock_guard<std::mutex> lock(*update_mtx_);
        const float* inp = input.data();
        
        float eff_lr = lr_ * std::max(0.01f, std::min(reward_mod, 3.f));
        
        auto apply_plasticity = [&](int node_idx, float scale) {
            auto& w = nodes_[node_idx].weights;
            for (int j = 0; j < n_dims; j++) {
                if (std::abs(inp[j]) < 1e-4f) continue; // Sparse update
                w[j] += eff_lr * scale * (inp[j] - w[j]);
            }
            nodes_[node_idx].hits += 1.0f;
            nodes_[node_idx].last_visited = step_;
        };
        
        // 1. Update BMU (Full plasticity)
        apply_plasticity(bmu, 1.0f);
        entry_point_ = bmu; // Shift entry point to active regions
        
        // 2. Update Direct Lateral Neighbors (Partial plasticity)
        for (int neighbor : nodes_[bmu].neighbors) {
            apply_plasticity(neighbor, 0.5f);
        }
        
        // 3. Dynamic Graph Rewiring (Hebbian Learning)
        // Every 100 hits, the BMU tries to connect to a random node that is physically close
        if (nodes_[bmu].hits > 100.0f) {
            nodes_[bmu].hits = 0.f;
            std::mt19937 rng(step_);
            std::uniform_int_distribution<int> rand_node(0, n_neurons - 1);
            
            int candidate = rand_node(rng);
            if (candidate != bmu) {
                float dist = vec_l2sq(nodes_[candidate].weights, nodes_[bmu].weights.data());
                if (dist < 1.0f) { // If they are semantically close
                    // Add edge if not full
                    if (nodes_[bmu].neighbors.size() < max_neighbors) {
                        nodes_[bmu].neighbors.push_back(candidate);
                    } else {
                        // Replace the oldest/least relevant neighbor
                        nodes_[bmu].neighbors[rng() % max_neighbors] = candidate;
                    }
                    
                    // Bidirectional link
                    if (nodes_[candidate].neighbors.size() < max_neighbors) {
                        nodes_[candidate].neighbors.push_back(bmu);
                    }
                }
            }
        }
        
        lr_ *= lr_decay_;
        step_++;
    }

    void prune_dead_branches(int max_age) {
        // Not strictly needed in a graph, but we could cull isolated nodes.
        // For now, nodes simply drift.
    }

    std::vector<float> neuron_weights(int i) const {
        if (i < 0 || i >= n_neurons)
            throw std::out_of_range("SOM::neuron_weights: out of range");
        return nodes_[i].weights;
    }

    int   step()   const noexcept { return step_;   }
    float lr()     const noexcept { return lr_;     }
    float radius() const noexcept { return 1.0f;    } // API compat

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SOM::save: cannot open " + path);
        f.write((const char*)&rows,          sizeof(int));
        f.write((const char*)&cols,          sizeof(int));
        f.write((const char*)&n_neurons,     sizeof(int));
        f.write((const char*)&n_dims,        sizeof(int));
        f.write((const char*)&max_neighbors, sizeof(int));
        f.write((const char*)&entry_point_,  sizeof(int));
        f.write((const char*)&lr_,           sizeof(float));
        f.write((const char*)&lr_decay_,     sizeof(float));
        f.write((const char*)&step_,         sizeof(int));
        
        for (const auto& node : nodes_) {
            f.write((const char*)node.weights.data(), n_dims * sizeof(float));
            f.write((const char*)&node.hits, sizeof(float));
            f.write((const char*)&node.last_visited, sizeof(int));
            
            int num_neighbors = (int)node.neighbors.size();
            f.write((const char*)&num_neighbors, sizeof(int));
            if (num_neighbors > 0) {
                f.write((const char*)node.neighbors.data(), num_neighbors * sizeof(int));
            }
        }
        
        tlb_.save(path + ".tlb");
    }

    static SOM load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SOM::load: cannot open " + path);
        SOM s;
        f.read((char*)&s.rows,          sizeof(int));
        f.read((char*)&s.cols,          sizeof(int));
        f.read((char*)&s.n_neurons,     sizeof(int));
        f.read((char*)&s.n_dims,        sizeof(int));
        f.read((char*)&s.max_neighbors, sizeof(int));
        f.read((char*)&s.entry_point_,  sizeof(int));
        f.read((char*)&s.lr_,           sizeof(float));
        f.read((char*)&s.lr_decay_,     sizeof(float));
        f.read((char*)&s.step_,         sizeof(int));
        
        s.nodes_.resize(s.n_neurons);
        for (int i = 0; i < s.n_neurons; i++) {
            s.nodes_[i].weights.resize(s.n_dims);
            f.read((char*)s.nodes_[i].weights.data(), s.n_dims * sizeof(float));
            f.read((char*)&s.nodes_[i].hits, sizeof(float));
            f.read((char*)&s.nodes_[i].last_visited, sizeof(int));
            
            int num_neighbors;
            f.read((char*)&num_neighbors, sizeof(int));
            if (num_neighbors > 0) {
                s.nodes_[i].neighbors.resize(num_neighbors);
                f.read((char*)s.nodes_[i].neighbors.data(), num_neighbors * sizeof(int));
            }
        }
        
        s.update_mtx_ = std::make_unique<std::mutex>();
        
        try {
            s.tlb_ = CognitiveTLB::load(path + ".tlb");
        } catch (...) {
            s.tlb_ = CognitiveTLB(s.n_dims, 64, 42);
        }
        
        return s;
    }

    void expand_dims(int new_dims) {
        std::lock_guard<std::mutex> lock(*update_mtx_);
        if (new_dims <= n_dims) return;
        for (auto& node : nodes_) {
            node.weights.resize(new_dims, 0.f);
        }
        n_dims = new_dims;
    }
};

} // namespace brain2
