#pragma once
/*
 * som.hpp — Hierarchical Grid Self-Organizing Map (HSOM)
 *
 * Implements an O(log N) tree traversal across multiple 2D sub-grids.
 * Overcomes flat SOM limits while preserving continuous topological manifolds.
 * Sub-grids spawn dynamically when a neuron receives excessive hits.
 * Output activation maintains the original fixed-dimensional API by leaving
 * unused/unvisited neurons dormant (0.0).
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

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace brain2 {

struct SubGrid {
    int   start_idx;     // offset in the global pool array
    int   rows;
    int   cols;
    int   parent_idx;    // index of parent neuron (-1 for root)
    std::vector<int> children; // indices of child subgrids
    
    SubGrid(int s_idx, int r, int c, int p_idx) 
      : start_idx(s_idx), rows(r), cols(c), parent_idx(p_idx) {
        children.resize(r * c, -1);
    }
};

class SOM {
public:
    int rows, cols, n_neurons, n_dims;
    int sub_rows, sub_cols;

private:
    std::vector<float>           weights_;     // Fixed global capacity pool
    std::vector<float>           hits_;        // Hit counts per neuron
    std::vector<SubGrid>         grids_;       // Hierarchy of grids
    int                          next_alloc_;  // Pointer for pool allocation
    float                        lr_, radius_, lr_decay_, radius_decay_;
    int                          step_;
    std::unique_ptr<std::mutex>  update_mtx_;

    inline float l2sq(const float* __restrict__ a,
                      const float* __restrict__ b) const noexcept {
        float s = 0.f;
        for (int i = 0; i < n_dims; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    // O(1) Search within a single 4x4 sub-grid
    int find_bmu_in_grid(int grid_idx, const float* inp, float& out_dist) const {
        const auto& g = grids_[grid_idx];
        int n_local = g.rows * g.cols;
        float best_d = std::numeric_limits<float>::max();
        int best_local = 0;

        for (int i = 0; i < n_local; i++) {
            float d = l2sq(inp, weights_.data() + (size_t)(g.start_idx + i) * n_dims);
            if (d < best_d) { best_d = d; best_local = i; }
        }
        out_dist = best_d;
        return best_local;
    }

public:
    SOM() : rows(0), cols(0), n_neurons(0), n_dims(0), sub_rows(4), sub_cols(4),
            next_alloc_(0), lr_(0), radius_(0), lr_decay_(0), radius_decay_(0), step_(0),
            update_mtx_(std::make_unique<std::mutex>()) {}

    SOM(int rows, int cols, int n_dims,
        float init_lr      = 0.15f,
        float lr_decay     = 0.9998f,
        float radius_decay = 0.9999f,
        unsigned seed      = 42)
        : rows(rows), cols(cols), n_neurons(rows * cols), n_dims(n_dims),
          sub_rows(4), sub_cols(4), // 4x4 default sub-grids
          next_alloc_(0),
          lr_(init_lr),
          radius_(float(std::max(sub_rows, sub_cols)) / 2.f),
          lr_decay_(lr_decay), radius_decay_(radius_decay), step_(0),
          update_mtx_(std::make_unique<std::mutex>())
    {
        weights_.resize(size_t(n_neurons) * n_dims, 0.f);
        hits_.resize(n_neurons, 0.f);
        
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.f, 0.3f);
        for (auto& w : weights_) w = dist(rng);
        
        // Bootstrap root grid
        grids_.emplace_back(next_alloc_, sub_rows, sub_cols, -1);
        next_alloc_ += sub_rows * sub_cols;
    }

    SOM(SOM&&)            = default;
    SOM& operator=(SOM&&) = default;
    SOM(const SOM&)       = delete;
    SOM& operator=(const SOM&) = delete;

    // Fast O(log N) search using hierarchy
    int find_bmu(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::find_bmu: dim mismatch");
            
        const float* inp = input.data();
        int curr_grid = 0; // Root
        int global_bmu = -1;
        
        while (true) {
            float dist;
            int local_bmu = find_bmu_in_grid(curr_grid, inp, dist);
            global_bmu = grids_[curr_grid].start_idx + local_bmu;
            
            int child_grid = grids_[curr_grid].children[local_bmu];
            if (child_grid != -1) {
                curr_grid = child_grid; // Descend tree
            } else {
                break; // Hit leaf
            }
        }
        return global_bmu;
    }

    // Sparse hierarchical activation map
    std::vector<float> activation_map(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::activation_map: dim mismatch");
            
        const float* inp = input.data();
        std::vector<float> dists(n_neurons, std::numeric_limits<float>::max());
        std::vector<float> acts(n_neurons, 0.f);
        
        int curr_grid = 0;
        
        // Traverse and calculate distance only for visited sub-grids
        while (true) {
            float d;
            int local_bmu = find_bmu_in_grid(curr_grid, inp, d);
            int child_grid = grids_[curr_grid].children[local_bmu];
            
            int s_idx = grids_[curr_grid].start_idx;
            int n_local = grids_[curr_grid].rows * grids_[curr_grid].cols;
            for (int i = 0; i < n_local; i++) {
                dists[s_idx + i] = l2sq(inp, weights_.data() + size_t(s_idx + i) * n_dims);
            }
            
            if (child_grid != -1) curr_grid = child_grid;
            else break;
        }
        
        float mn = std::numeric_limits<float>::max();
        for (float d : dists) if (d < mn) mn = d;
        
        std::vector<float> deltas;
        deltas.reserve(n_neurons);
        for (float d : dists) {
            if (d != std::numeric_limits<float>::max()) {
                deltas.push_back(d - mn);
            }
        }
        
        size_t kth = std::max<size_t>(1, deltas.size() / 5); 
        std::nth_element(deltas.begin(), deltas.begin() + kth, deltas.end());
        float sigma = std::max(deltas[kth], 1e-6f);
        
        // Only traversed neurons get non-zero activation
        for (int i = 0; i < n_neurons; i++) {
            if (dists[i] != std::numeric_limits<float>::max()) {
                acts[i] = std::exp(-(dists[i] - mn) / sigma);
            }
        }
        return acts;
    }

    inline float grid_dist(int local_i, int local_j) const noexcept {
        float dr = float(local_i / sub_cols) - float(local_j / sub_cols);
        float dc = float(local_i % sub_cols) - float(local_j % sub_cols);
        return std::sqrt(dr * dr + dc * dc);
    }

    // Update weights and dynamically spawn sub-grids
    void update(const std::vector<float>& input, int bmu, float reward_mod = 1.f) {
        std::lock_guard<std::mutex> lock(*update_mtx_);
        const float* inp = input.data();
        
        int curr_grid = 0;
        int leaf_grid = -1;
        int leaf_local_bmu = -1;
        
        // 1. Update the entire active branch
        while (curr_grid != -1) {
            float dist;
            int local_bmu = find_bmu_in_grid(curr_grid, inp, dist);
            
            float eff_lr = lr_ * std::max(0.01f, std::min(reward_mod, 3.f));
            float r2 = radius_ * radius_ * 2.f;
            int s_idx = grids_[curr_grid].start_idx;
            int n_local = grids_[curr_grid].rows * grids_[curr_grid].cols;
            
            for (int i = 0; i < n_local; i++) {
                float d = grid_dist(i, local_bmu);
                float h = std::exp(-d * d / r2);
                if (h < 1e-4f) continue;
                
                float* w = weights_.data() + size_t(s_idx + i) * n_dims;
                float sc = eff_lr * h;
                for (int j = 0; j < n_dims; j++) {
                    w[j] += sc * (inp[j] - w[j]);
                }
            }
            
            int child_grid = grids_[curr_grid].children[local_bmu];
            if (child_grid == -1) {
                leaf_grid = curr_grid;
                leaf_local_bmu = local_bmu;
            }
            curr_grid = child_grid;
        }
        
        // 2. Spawn logic: if a leaf neuron gets crowded, it branches
        int global_bmu = grids_[leaf_grid].start_idx + leaf_local_bmu;
        hits_[global_bmu] += 1.0f;
        
        if (hits_[global_bmu] > 50.0f) { // Spawning threshold
            int needed = sub_rows * sub_cols;
            if (next_alloc_ + needed <= n_neurons) {
                int new_grid_idx = (int)grids_.size();
                grids_.emplace_back(next_alloc_, sub_rows, sub_cols, global_bmu);
                grids_[leaf_grid].children[leaf_local_bmu] = new_grid_idx;
                
                std::mt19937 rng(step_);
                std::normal_distribution<float> noise(0.f, 0.05f);
                float* p_w = weights_.data() + size_t(global_bmu) * n_dims;
                for (int i = 0; i < needed; i++) {
                    float* c_w = weights_.data() + size_t(next_alloc_ + i) * n_dims;
                    for (int j = 0; j < n_dims; j++) {
                        c_w[j] = p_w[j] + noise(rng);
                    }
                }
                next_alloc_ += needed;
                hits_[global_bmu] = 0.f; 
            }
        }
        
        lr_     *= lr_decay_;
        radius_ *= radius_decay_;
        step_++;
    }

    std::vector<float> neuron_weights(int i) const {
        if (i < 0 || i >= n_neurons)
            throw std::out_of_range("SOM::neuron_weights: out of range");
        auto b = weights_.begin() + size_t(i) * n_dims;
        return std::vector<float>(b, b + n_dims);
    }

    int   step()   const noexcept { return step_;   }
    float lr()     const noexcept { return lr_;     }
    float radius() const noexcept { return radius_; }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SOM::save: cannot open " + path);
        f.write((const char*)&rows,          sizeof(int));
        f.write((const char*)&cols,          sizeof(int));
        f.write((const char*)&n_dims,        sizeof(int));
        f.write((const char*)&sub_rows,      sizeof(int));
        f.write((const char*)&sub_cols,      sizeof(int));
        f.write((const char*)&next_alloc_,   sizeof(int));
        f.write((const char*)&lr_,           sizeof(float));
        f.write((const char*)&radius_,       sizeof(float));
        f.write((const char*)&lr_decay_,     sizeof(float));
        f.write((const char*)&radius_decay_, sizeof(float));
        f.write((const char*)&step_,         sizeof(int));
        f.write((const char*)weights_.data(),
                (std::streamsize)(weights_.size() * sizeof(float)));
        f.write((const char*)hits_.data(),
                (std::streamsize)(hits_.size() * sizeof(float)));
                
        int n_grids = (int)grids_.size();
        f.write((const char*)&n_grids, sizeof(int));
        for (const auto& g : grids_) {
            f.write((const char*)&g.start_idx,  sizeof(int));
            f.write((const char*)&g.rows,       sizeof(int));
            f.write((const char*)&g.cols,       sizeof(int));
            f.write((const char*)&g.parent_idx, sizeof(int));
            int nc = (int)g.children.size();
            f.write((const char*)&nc, sizeof(int));
            f.write((const char*)g.children.data(), nc * sizeof(int));
        }
    }

    static SOM load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SOM::load: cannot open " + path);
        SOM s;
        f.read((char*)&s.rows,          sizeof(int));
        f.read((char*)&s.cols,          sizeof(int));
        f.read((char*)&s.n_dims,        sizeof(int));
        f.read((char*)&s.sub_rows,      sizeof(int));
        f.read((char*)&s.sub_cols,      sizeof(int));
        f.read((char*)&s.next_alloc_,   sizeof(int));
        f.read((char*)&s.lr_,           sizeof(float));
        f.read((char*)&s.radius_,       sizeof(float));
        f.read((char*)&s.lr_decay_,     sizeof(float));
        f.read((char*)&s.radius_decay_, sizeof(float));
        f.read((char*)&s.step_,         sizeof(int));
        
        s.n_neurons = s.rows * s.cols;
        s.weights_.resize(size_t(s.n_neurons) * s.n_dims);
        s.hits_.resize(s.n_neurons);
        
        f.read((char*)s.weights_.data(),
               (std::streamsize)(s.weights_.size() * sizeof(float)));
        f.read((char*)s.hits_.data(),
               (std::streamsize)(s.hits_.size() * sizeof(float)));
               
        int n_grids;
        f.read((char*)&n_grids, sizeof(int));
        for (int i = 0; i < n_grids; i++) {
            int si, r, c, pi, nc;
            f.read((char*)&si, sizeof(int));
            f.read((char*)&r, sizeof(int));
            f.read((char*)&c, sizeof(int));
            f.read((char*)&pi, sizeof(int));
            f.read((char*)&nc, sizeof(int));
            
            SubGrid g(si, r, c, pi);
            g.children.resize(nc);
            f.read((char*)g.children.data(), nc * sizeof(int));
            s.grids_.push_back(std::move(g));
        }
        s.update_mtx_ = std::make_unique<std::mutex>();
        return s;
    }
};

} // namespace brain2
