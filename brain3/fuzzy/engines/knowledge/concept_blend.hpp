#pragma once
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <tuple>
#include <algorithm>

namespace brain2 {
namespace knowledge {

using Vector = std::vector<float>;

inline float dist(const Vector& a, const Vector& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

inline std::pair<std::string, float> nearest(
    const Vector& v, 
    const std::map<std::string, Vector>& concepts, 
    const std::vector<std::string>& exclude = {}) {
    
    std::string best_name = "";
    float best_dist = std::numeric_limits<float>::infinity();
    
    for (const auto& kv : concepts) {
        if (std::find(exclude.begin(), exclude.end(), kv.first) != exclude.end()) continue;
        float d = dist(v, kv.second);
        if (d < best_dist) {
            best_dist = d;
            best_name = kv.first;
        }
    }
    return {best_name, best_dist};
}

inline Vector blend(const std::string& a_name, const std::string& b_name, 
                    const std::map<std::string, Vector>& concepts, 
                    const std::string& mode = "salient") {
    
    const Vector& a = concepts.at(a_name);
    const Vector& b = concepts.at(b_name);
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

inline std::tuple<bool, std::string, float> verify_novel(const Vector& v, const std::map<std::string, Vector>& concepts, float radius) {
    auto [near_name, d] = nearest(v, concepts);
    return {d > radius, near_name, d};
}

struct ProposeResult {
    Vector vector;
    bool novel;
    std::string nearest;
    float distance;
};

inline ProposeResult propose(const std::string& a_name, const std::string& b_name, 
                             const std::map<std::string, Vector>& concepts, 
                             float radius, const std::string& mode = "salient") {
    
    Vector v = blend(a_name, b_name, concepts, mode);
    auto [novel, near_name, d] = verify_novel(v, concepts, radius);
    return {v, novel, near_name, d};
}

} // namespace knowledge
} // namespace brain2
