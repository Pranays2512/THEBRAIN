#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class DecisionTree {
public:
    struct Node {
        bool is_leaf;
        int feature_index;
        std::vector<double> dist;
        std::unique_ptr<Node> left;  // mask (X > 0.5)
        std::unique_ptr<Node> right; // ~mask (X <= 0.5)

        Node() : is_leaf(false), feature_index(-1) {}
    };

    int n_ops;
    int max_depth;
    int min_samples;
    std::unique_ptr<Node> root;

    DecisionTree(int n_ops, int max_depth = 10, int min_samples = 15)
        : n_ops(n_ops), max_depth(max_depth), min_samples(min_samples) {}

    std::vector<double> _dist(const std::vector<int>& y, const std::vector<int>& indices) {
        std::vector<double> c(n_ops, 0.1); // smoothing
        for (int idx : indices) {
            c[y[idx]] += 1.0;
        }
        double sum = 0.0;
        for (double val : c) sum += val;
        for (double& val : c) val /= sum;
        return c;
    }

    double _gini(const std::vector<int>& y, const std::vector<int>& indices) {
        std::vector<double> d = _dist(y, indices);
        double sum_sq = 0.0;
        for (double val : d) sum_sq += val * val;
        return 1.0 - sum_sq;
    }

    std::unique_ptr<Node> _build(const float* X_data, int n_features, const std::vector<int>& y, const std::vector<int>& indices, int depth) {
        auto node = std::make_unique<Node>();

        bool all_same = true;
        if (!indices.empty()) {
            int first_val = y[indices[0]];
            for (size_t i = 1; i < indices.size(); ++i) {
                if (y[indices[i]] != first_val) {
                    all_same = false;
                    break;
                }
            }
        }

        if (depth >= max_depth || indices.size() < (size_t)min_samples || all_same) {
            node->is_leaf = true;
            node->dist = _dist(y, indices);
            return node;
        }

        double base_gini = _gini(y, indices);
        double base = base_gini * indices.size();
        
        double best_gain = -1.0;
        int best_feature = -1;
        std::vector<int> best_left_indices;
        std::vector<int> best_right_indices;

        for (int f = 0; f < n_features; ++f) {
            std::vector<int> left_indices;
            std::vector<int> right_indices;
            left_indices.reserve(indices.size());
            right_indices.reserve(indices.size());

            for (int idx : indices) {
                if (X_data[idx * n_features + f] > 0.5f) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }

            if (left_indices.empty() || right_indices.empty()) continue;

            double cost = _gini(y, left_indices) * left_indices.size() + 
                          _gini(y, right_indices) * right_indices.size();
            double gain = base - cost;

            if (gain > best_gain + 1e-12) { // tie-breaker margin to match Python's 'gain > best[0]' precisely
                best_gain = gain;
                best_feature = f;
                best_left_indices = std::move(left_indices);
                best_right_indices = std::move(right_indices);
            }
        }

        if (best_gain <= 1e-9) {
            node->is_leaf = true;
            node->dist = _dist(y, indices);
            return node;
        }

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->left = _build(X_data, n_features, y, best_left_indices, depth + 1);
        node->right = _build(X_data, n_features, y, best_right_indices, depth + 1);
        return node;
    }

    void fit(py::array_t<float> X, py::array_t<int> y) {
        py::buffer_info X_info = X.request();
        py::buffer_info y_info = y.request();

        if (X_info.ndim != 2 || y_info.ndim != 1) {
            throw std::runtime_error("X must be 2D and y must be 1D");
        }
        if (X_info.shape[0] != y_info.shape[0]) {
            throw std::runtime_error("X and y must have the same number of samples");
        }

        int n_samples = X_info.shape[0];
        int n_features = X_info.shape[1];

        const float* X_data = static_cast<const float*>(X_info.ptr);
        const int* y_data = static_cast<const int*>(y_info.ptr);

        std::vector<int> y_vec(y_data, y_data + n_samples);
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        root = _build(X_data, n_features, y_vec, indices, 0);
    }

    std::vector<double> predict_dist(py::array_t<float> x) {
        py::buffer_info x_info = x.request();
        if (x_info.ndim != 1) {
            throw std::runtime_error("x must be 1D");
        }

        const float* x_data = static_cast<const float*>(x_info.ptr);
        Node* node = root.get();

        if (!node) {
            throw std::runtime_error("Tree is not fitted yet");
        }

        while (!node->is_leaf) {
            if (x_data[node->feature_index] > 0.5f) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        return node->dist;
    }
};
