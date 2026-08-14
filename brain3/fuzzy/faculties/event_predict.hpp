#pragma once
#include <map>
#include <string>
#include <vector>

namespace brain2 {
namespace faculties {

// Probabilistic sequence modeling
class EventPredictor {
private:
    std::map<std::string, std::map<std::string, int>> transitions;
    std::map<std::string, int> counts;

public:
    void observe(const std::string& state_a, const std::string& state_b) {
        transitions[state_a][state_b]++;
        counts[state_a]++;
    }
    
    std::string predict_next(const std::string& state) {
        if (!transitions.count(state)) return "";
        
        std::string best = "";
        int max_c = -1;
        for (const auto& [nxt, c] : transitions[state]) {
            if (c > max_c) {
                max_c = c;
                best = nxt;
            }
        }
        return best;
    }
};

} // namespace faculties
} // namespace brain2
