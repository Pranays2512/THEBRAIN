#pragma once
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <stdexcept>
#include <iostream>

#include "grounding.hpp"

namespace brain2 {
namespace grounding {

inline Vector blend_centroid(const std::map<std::string, Vector>& centroids, const std::string& a_name, const std::string& b_name, const std::string& mode = "salient") {
    if (centroids.count(a_name) == 0 || centroids.count(b_name) == 0) {
        throw std::runtime_error("Centroid not found");
    }
    const Vector& a = centroids.at(a_name);
    const Vector& b = centroids.at(b_name);
    
    Vector res(a.size());
    if (mode == "mid") {
        for (size_t i = 0; i < a.size(); ++i) res[i] = (a[i] + b[i]) / 2.0f;
    } else { // salient
        for (size_t i = 0; i < a.size(); ++i) {
            res[i] = (std::abs(a[i]) >= std::abs(b[i])) ? a[i] : b[i];
        }
    }
    return res;
}

// Returns {novel, nearest, sim}
inline std::tuple<bool, std::string, float> verify_novel(const std::map<std::string, Vector>& centroids, const Vector& c, float sim_floor) {
    if (centroids.empty()) return {true, "", 0.0f};
    
    std::string nearest = "";
    float best_sim = -2.0f;
    
    for (const auto& kv : centroids) {
        float sim = cosine_similarity(c, kv.second);
        if (sim > best_sim) {
            best_sim = sim;
            nearest = kv.first;
        }
    }
    
    bool novel = best_sim < sim_floor;
    return {novel, nearest, best_sim};
}

inline Vector create_chimera_raw(const Vector& a_raw, const Vector& b_raw) {
    Vector res(a_raw.size());
    size_t h = a_raw.size() / 2;
    for (size_t i = 0; i < h; ++i) res[i] = a_raw[i];
    for (size_t i = h; i < a_raw.size(); ++i) res[i] = b_raw[i];
    return res;
}

} // namespace grounding
} // namespace brain2
