#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <map>

namespace brain2 {
namespace reasoning {

// Very simple Gaussian elimination for linear least squares (normal equations)
inline std::vector<double> lstsq(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (X.empty()) return {};
    int n = X.size();
    int m = X[0].size();
    
    // X^T X
    std::vector<std::vector<double>> XtX(m, std::vector<double>(m, 0.0));
    // X^T y
    std::vector<double> Xty(m, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Xty[j] += X[i][j] * y[i];
            for (int k = 0; k < m; k++) {
                XtX[j][k] += X[i][j] * X[i][k];
            }
        }
    }

    // Add small ridge to avoid singularity
    for (int i = 0; i < m; i++) XtX[i][i] += 1e-6;

    // Gauss-Jordan elimination on [XtX | Xty]
    for (int i = 0; i < m; i++) {
        // pivot
        double max_val = std::abs(XtX[i][i]);
        int pivot = i;
        for (int j = i + 1; j < m; j++) {
            if (std::abs(XtX[j][i]) > max_val) {
                max_val = std::abs(XtX[j][i]);
                pivot = j;
            }
        }
        std::swap(XtX[i], XtX[pivot]);
        std::swap(Xty[i], Xty[pivot]);

        double div = XtX[i][i];
        for (int j = i; j < m; j++) XtX[i][j] /= div;
        Xty[i] /= div;

        for (int j = 0; j < m; j++) {
            if (i != j) {
                double sub = XtX[j][i];
                for (int k = i; k < m; k++) XtX[j][k] -= sub * XtX[i][k];
                Xty[j] -= sub * Xty[i];
            }
        }
    }
    return Xty;
}

class LearnedHeuristic {
private:
    std::function<std::vector<double>(const std::vector<int>&)> feature_fn;
    std::vector<double> weights;
    bool trained = false;

public:
    LearnedHeuristic(std::function<std::vector<double>(const std::vector<int>&)> feature_fn, 
                     std::vector<double> w = {}) : feature_fn(feature_fn), weights(w) {
        if (!weights.empty()) trained = true;
    }

    void train(const std::vector<std::pair<std::vector<int>, double>>& examples) {
        if (examples.empty()) throw std::runtime_error("no training examples");
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        for (const auto& ex : examples) {
            X.push_back(feature_fn(ex.first));
            y.push_back(ex.second);
        }
        weights = lstsq(X, y);
        trained = true;
    }

    double operator()(const std::vector<int>& state) const {
        if (!trained) throw std::runtime_error("heuristic used before train()");
        std::vector<double> feats = feature_fn(state);
        double est = 0.0;
        for (size_t i = 0; i < feats.size() && i < weights.size(); i++) {
            est += feats[i] * weights[i];
        }
        return std::max(0.0, est);
    }
};

}}
