#pragma once
#include <string>
#include <vector>
#include <cmath>

namespace brain2 {
namespace faculties {

class CuriosityBridge {
public:
    double calculate_similarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size() || a.empty()) return 0.0;
        double dot = 0.0, mag_a = 0.0, mag_b = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            dot += a[i] * b[i];
            mag_a += a[i] * a[i];
            mag_b += b[i] * b[i];
        }
        if (mag_a == 0 || mag_b == 0) return 0.0;
        return dot / (std::sqrt(mag_a) * std::sqrt(mag_b));
    }
};

} // namespace faculties
} // namespace brain2
