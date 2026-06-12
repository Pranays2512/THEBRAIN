#pragma once
/*
 * som.hpp — Navigable Small World (NSW) Graph SOM
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

class SOM {
public:
    int rows, cols;
    int n_neurons, n_dims;
    int max_neighbors;

private:
    std::vector<float> weights_;
    std::vector<std::vector<int>> neighbors_;
    std::vector<float> hits_;
    std::vector<int> last_visited_;

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
        weights_.resize(n_neurons * n_dims);
        neighbors_.resize(n_neurons);
        hits_.resize(n_neurons, 0.f);
        last_visited_.resize(n_neurons, 0);

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.f, 0.3f);
        std::uniform_int_distribution<int> rand_node(0, n_neurons - 1);

        for (int i = 0; i < n_neurons; i++) {
            for (int j = 0; j < n_dims; j++) {
                weights_[i * n_dims + j] = dist(rng);
            }
            for (int k = 0; k < max_neighbors; k++) {
                int neighbor = rand_node(rng);
                if (neighbor != i) {
                    neighbors_[i].push_back(neighbor);
                }
            }
        }
    }

    SOM(SOM&&)            = default;
    SOM& operator=(SOM&&) = default;
    SOM(const SOM&)       = delete;
    SOM& operator=(const SOM&) = delete;

    // Fast O(log N) Greedy Graph Search -- SWAPPED TO BRUTE FORCE Linear Scan (Correctness first)
    int find_bmu(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::find_bmu: dim mismatch");
            
        const float* inp = input.data();
        int best_node = 0;
        float best_dist = std::numeric_limits<float>::max();
        
        #ifdef _OPENMP
        #pragma omp parallel
        {
            int local_best_node = 0;
            float local_best_dist = std::numeric_limits<float>::max();
            
            #pragma omp for nowait
            for (int i = 0; i < n_neurons; i++) {
                float d = l2sq(&weights_[i * n_dims], inp);
                if (d < local_best_dist) {
                    local_best_dist = d;
                    local_best_node = i;
                }
            }
            
            #pragma omp critical
            {
                if (local_best_dist < best_dist) {
                    best_dist = local_best_dist;
                    best_node = local_best_node;
                }
            }
        }
        #else
        for (int i = 0; i < n_neurons; i++) {
            float d = l2sq(&weights_[i * n_dims], inp);
            if (d < best_dist) {
                best_dist = d;
                best_node = i;
            }
        }
        #endif
        
        return best_node;
    }

    // Graph-based sparse activation
    std::vector<float> activation_map(const std::vector<float>& input) const {
        std::vector<float> acts(n_neurons, 0.f);
        int bmu = find_bmu(input);
        
        // Activate BMU
        acts[bmu] = 1.0f;
        
        // Activate neighbors (decaying outward)
        const float* inp = input.data();
        for (int neighbor : neighbors_[bmu]) {
            float d = l2sq(&weights_[neighbor * n_dims], inp);
            acts[neighbor] = std::exp(-d * 2.0f); // Fast decay
        }
        
        return acts;
    }

    // Single-pass: returns BMU and activation map using one distance scan.
    // Eliminates the triple scan (find_bmu + activation_map + update each recompute distances).
    std::pair<int, std::vector<float>> find_bmu_and_activate(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::find_bmu_and_activate: dim mismatch");

        const float* inp = input.data();
        int best_node = 0;
        float best_dist = std::numeric_limits<float>::max();

        // One full distance scan — find BMU
        for (int i = 0; i < n_neurons; i++) {
            float d = l2sq(&weights_[i * n_dims], inp);
            if (d < best_dist) {
                best_dist = d;
                best_node = i;
            }
        }

        // Build activation map using BMU neighborhood (no second full scan)
        std::vector<float> acts(n_neurons, 0.f);
        acts[best_node] = 1.0f;
        for (int neighbor : neighbors_[best_node]) {
            float d = l2sq(&weights_[neighbor * n_dims], inp);
            acts[neighbor] = std::exp(-d * 2.0f);
        }

        return {best_node, std::move(acts)};
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
            float* w = &weights_[node_idx * n_dims];
            for (int j = 0; j < n_dims; j++) {
                if (std::abs(inp[j]) < 1e-4f) continue; // Sparse update
                w[j] += eff_lr * scale * (inp[j] - w[j]);
            }
            hits_[node_idx] += 1.0f;
            last_visited_[node_idx] = step_;
        };
        
        // 1. Update BMU (Full plasticity)
        apply_plasticity(bmu, 1.0f);
        entry_point_ = bmu; // Shift entry point to active regions
        
        // 2. Update Direct Lateral Neighbors (Partial plasticity)
        for (int neighbor : neighbors_[bmu]) {
            apply_plasticity(neighbor, 0.5f);
        }
        
        // 3. Dynamic Graph Rewiring (Hebbian Learning)
        // Every 100 hits, the BMU tries to connect to a random node that is physically close
        if (hits_[bmu] > 100.0f) {
            hits_[bmu] = 0.f;
            std::mt19937 rng(step_);
            std::uniform_int_distribution<int> rand_node(0, n_neurons - 1);
            
            int candidate = rand_node(rng);
            if (candidate != bmu) {
                float dist = l2sq(&weights_[candidate * n_dims], &weights_[bmu * n_dims]);
                if (dist < 1.0f) { // If they are semantically close
                    // Add edge if not full
                    if (neighbors_[bmu].size() < (size_t)max_neighbors) {
                        neighbors_[bmu].push_back(candidate);
                    } else {
                        // Replace the oldest/least relevant neighbor
                        neighbors_[bmu][rng() % max_neighbors] = candidate;
                    }
                    
                    // Bidirectional link
                    if (neighbors_[candidate].size() < (size_t)max_neighbors) {
                        neighbors_[candidate].push_back(bmu);
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
        return std::vector<float>(weights_.begin() + i * n_dims, weights_.begin() + (i + 1) * n_dims);
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
        
        f.write((const char*)weights_.data(), weights_.size() * sizeof(float));
        f.write((const char*)hits_.data(), hits_.size() * sizeof(float));
        f.write((const char*)last_visited_.data(), last_visited_.size() * sizeof(int));
        
        for (int i = 0; i < n_neurons; i++) {
            int num_neighbors = (int)neighbors_[i].size();
            f.write((const char*)&num_neighbors, sizeof(int));
            if (num_neighbors > 0) {
                f.write((const char*)neighbors_[i].data(), num_neighbors * sizeof(int));
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
        
        s.weights_.resize(s.n_neurons * s.n_dims);
        f.read((char*)s.weights_.data(), s.weights_.size() * sizeof(float));
        
        s.hits_.resize(s.n_neurons);
        f.read((char*)s.hits_.data(), s.hits_.size() * sizeof(float));
        
        s.last_visited_.resize(s.n_neurons);
        f.read((char*)s.last_visited_.data(), s.last_visited_.size() * sizeof(int));
        
        s.neighbors_.resize(s.n_neurons);
        for (int i = 0; i < s.n_neurons; i++) {
            int num_neighbors;
            f.read((char*)&num_neighbors, sizeof(int));
            if (num_neighbors > 0) {
                s.neighbors_[i].resize(num_neighbors);
                f.read((char*)s.neighbors_[i].data(), num_neighbors * sizeof(int));
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
        std::vector<float> new_weights(n_neurons * new_dims, 0.f);
        for (int i = 0; i < n_neurons; i++) {
            for (int j = 0; j < n_dims; j++) {
                new_weights[i * new_dims + j] = weights_[i * n_dims + j];
            }
        }
        weights_ = std::move(new_weights);
        n_dims = new_dims;
    }
};

} // namespace brain2
