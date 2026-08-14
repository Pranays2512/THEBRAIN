#pragma once
#include <vector>
#include <cmath>

namespace brain2 {
namespace faculties {

// Statistical fuzzy feature learner
class FeatureLearner {
public:
    double learning_rate = 0.01;
    
    // Very simple online PCA / feature extraction stub
    std::vector<double> learn_feature(const std::vector<double>& input, std::vector<double>& weights) {
        if (weights.size() != input.size()) {
            weights = std::vector<double>(input.size(), 0.1);
        }
        
        double activation = 0.0;
        for (size_t i = 0; i < input.size(); i++) {
            activation += input[i] * weights[i];
        }
        
        // Oja's rule update
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] += learning_rate * activation * (input[i] - activation * weights[i]);
        }
        
        return weights;
    }
};

} // namespace faculties
} // namespace brain2
