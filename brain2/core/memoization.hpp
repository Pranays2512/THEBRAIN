#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <functional>

// Global Memoization Cache for the Brain
class MemoizationCache {
public:
    // Caches vector results (e.g. BIND_QUERY results, PUCT goal traces)
    std::unordered_map<std::string, std::vector<float>> vec_cache_;
    
    // Caches string/symbolic results (e.g. math calculations)
    std::unordered_map<std::string, std::string> str_cache_;

    // Performance metrics
    int hits_ = 0;
    int misses_ = 0;

    void clear() {
        vec_cache_.clear();
        str_cache_.clear();
        hits_ = 0;
        misses_ = 0;
    }

    // High-speed string caching
    bool has_str(const std::string& key) {
        if (str_cache_.find(key) != str_cache_.end()) {
            hits_++;
            return true;
        }
        misses_++;
        return false;
    }

    std::string get_str(const std::string& key) {
        return str_cache_[key];
    }

    void put_str(const std::string& key, const std::string& value) {
        str_cache_[key] = value;
    }

    // Vector caching
    bool has_vec(const std::string& key) {
        if (vec_cache_.find(key) != vec_cache_.end()) {
            hits_++;
            return true;
        }
        misses_++;
        return false;
    }

    std::vector<float> get_vec(const std::string& key) {
        return vec_cache_[key];
    }

    void put_vec(const std::string& key, const std::vector<float>& value) {
        vec_cache_[key] = value;
    }

    float hit_rate() const {
        if (hits_ + misses_ == 0) return 0.0f;
        return (float)hits_ / (hits_ + misses_);
    }
};
