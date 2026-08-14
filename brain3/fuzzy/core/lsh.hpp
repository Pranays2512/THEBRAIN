#pragma once

#include <vector>
#include <random>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include <fstream>
#include <stdexcept>

namespace brain2 {

/*
 * Cognitive TLB (Translation Lookaside Buffer) using Locality-Sensitive Hashing (LSH).
 * Maps a continuous "logical" thought vector to a "physical" BMU address in O(1) time.
 * Mimics an OS Page Table for the Neural Architecture.
 */
class CognitiveTLB {
private:
    int n_dims_;
    int n_bits_; // typically 64 for uint64_t
    std::vector<std::vector<float>> hyperplanes_;
    std::unordered_map<uint64_t, int> page_table_;
    std::unique_ptr<std::mutex> mtx_;

public:
    CognitiveTLB() : n_dims_(0), n_bits_(0), mtx_(std::make_unique<std::mutex>()) {}

    CognitiveTLB(int n_dims, int n_bits = 64, unsigned seed = 42)
        : n_dims_(n_dims), n_bits_(n_bits), mtx_(std::make_unique<std::mutex>()) {
        
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.f, 1.f);
        
        hyperplanes_.resize(n_bits, std::vector<float>(n_dims));
        for (int i = 0; i < n_bits; i++) {
            for (int j = 0; j < n_dims; j++) {
                hyperplanes_[i][j] = dist(rng);
            }
        }
    }

    CognitiveTLB(CognitiveTLB&&)            = default;
    CognitiveTLB& operator=(CognitiveTLB&&) = default;
    CognitiveTLB(const CognitiveTLB&)       = delete;
    CognitiveTLB& operator=(const CognitiveTLB&) = delete;

    // Project vector into a binary hash (the "Logical Address")
    uint64_t hash(const float* inp) const noexcept {
        uint64_t signature = 0;
        for (int i = 0; i < n_bits_; i++) {
            float dot = 0.f;
            for (int j = 0; j < n_dims_; j++) {
                dot += inp[j] * hyperplanes_[i][j];
            }
            if (dot > 0.f) {
                signature |= (1ULL << i);
            }
        }
        return signature;
    }

    // Try to find the physical address in O(1)
    int lookup(uint64_t logical_address) {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = page_table_.find(logical_address);
        if (it != page_table_.end()) {
            return it->second; // Cache Hit!
        }
        return -1; // Cache Miss (Page Fault)
    }

    // Update the Page Table
    void cache(uint64_t logical_address, int physical_bmu) {
        std::lock_guard<std::mutex> lock(*mtx_);
        // Overwrites are acceptable, acting like a cache eviction policy
        page_table_[logical_address] = physical_bmu;
    }
    
    // Save state to disk
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("CognitiveTLB::save: cannot open " + path);
        
        f.write((const char*)&n_dims_, sizeof(int));
        f.write((const char*)&n_bits_, sizeof(int));
        
        for (int i = 0; i < n_bits_; i++) {
            f.write((const char*)hyperplanes_[i].data(), n_dims_ * sizeof(float));
        }
        
        int n_entries = (int)page_table_.size();
        f.write((const char*)&n_entries, sizeof(int));
        for (const auto& pair : page_table_) {
            f.write((const char*)&pair.first, sizeof(uint64_t));
            f.write((const char*)&pair.second, sizeof(int));
        }
    }
    
    static CognitiveTLB load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("CognitiveTLB::load: cannot open " + path);
        
        CognitiveTLB tlb;
        f.read((char*)&tlb.n_dims_, sizeof(int));
        f.read((char*)&tlb.n_bits_, sizeof(int));
        
        tlb.hyperplanes_.resize(tlb.n_bits_, std::vector<float>(tlb.n_dims_));
        for (int i = 0; i < tlb.n_bits_; i++) {
            f.read((char*)tlb.hyperplanes_[i].data(), tlb.n_dims_ * sizeof(float));
        }
        
        int n_entries = 0;
        f.read((char*)&n_entries, sizeof(int));
        for (int i = 0; i < n_entries; i++) {
            uint64_t key;
            int val;
            f.read((char*)&key, sizeof(uint64_t));
            f.read((char*)&val, sizeof(int));
            tlb.page_table_[key] = val;
        }
        tlb.mtx_ = std::make_unique<std::mutex>();
        return tlb;
    }
};

} // namespace brain2
