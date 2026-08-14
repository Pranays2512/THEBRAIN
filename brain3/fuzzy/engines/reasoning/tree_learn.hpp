#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>

namespace brain2 {
namespace reasoning {

// Very simple least squares solver for learning heuristics
class TreeLearn {
public:
    std::vector<double> weights;

    // Features: num_samples x num_features
    // Labels: num_samples x 1 (the true remaining cost)
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        if (X.empty()) return;
        int n = X.size();
        int m = X[0].size();
        
        // Simple Gradient Descent since we don't have a full linear algebra library for exact lstsq
        weights.assign(m, 0.0);
        double lr = 0.01;
        int epochs = 1000;
        
        for (int ep = 0; ep < epochs; ++ep) {
            std::vector<double> grad(m, 0.0);
            for (int i = 0; i < n; ++i) {
                double pred = 0.0;
                for (int j = 0; j < m; ++j) pred += X[i][j] * weights[j];
                double err = pred - y[i];
                for (int j = 0; j < m; ++j) grad[j] += err * X[i][j];
            }
            for (int j = 0; j < m; ++j) weights[j] -= lr * (grad[j] / n);
        }
    }
    
    double predict(const std::vector<double>& features) const {
        if (weights.empty()) return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) sum += features[i] * weights[i];
        return std::max(0.0, sum); // Admissible heuristic should not be negative
    }
};

} // namespace reasoning
} // namespace brain2
