#pragma once
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "fuzzy/core/som.hpp"
#include "fuzzy/core/binding_memory.hpp"

namespace brain2 {
namespace grounding {

using Vector = std::vector<float>;

// Helper for dot product
inline float dot(const Vector& a, const Vector& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) sum += a[i] * b[i];
    return sum;
}

inline float norm(const Vector& a) {
    return std::sqrt(dot(a, a));
}

inline Vector normalize(const Vector& v) {
    float n = norm(v);
    if (n == 0.0f) return v;
    Vector res = v;
    for (auto& x : res) x /= n;
    return res;
}

inline float cosine_similarity(const Vector& a, const Vector& b) {
    float na = norm(a);
    float nb = norm(b);
    return (na > 0.0f && nb > 0.0f) ? (dot(a, b) / (na * nb)) : 0.0f;
}

// Generate deterministic unit vector for a string
inline Vector get_symbol_vector(const std::string& sym, int n_dims) {
    std::mt19937 gen(std::hash<std::string>{}(sym));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Vector v(n_dims);
    for (int i = 0; i < n_dims; ++i) v[i] = dist(gen);
    return normalize(v);
}

// Basic Matrix struct for least squares
struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& at(int r, int c) { return data[r * cols + c]; }
    const float& at(int r, int c) const { return data[r * cols + c]; }
    
    Matrix transpose() const {
        Matrix res(cols, rows);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                res.at(c, r) = at(r, c);
            }
        }
        return res;
    }
    
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) throw std::runtime_error("Matrix dims mismatch");
        Matrix res(rows, other.cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < other.cols; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < cols; ++k) sum += at(r, k) * other.at(k, c);
                res.at(r, c) = sum;
            }
        }
        return res;
    }
    
    // Gaussian elimination to solve Ax = b, where this is A (square matrix)
    Vector solve(Vector b) const {
        if (rows != cols || rows != (int)b.size()) throw std::runtime_error("Invalid dimensions for solve");
        Matrix A = *this;
        int n = rows;
        
        for (int i = 0; i < n; i++) {
            // Find pivot
            int max_row = i;
            for (int k = i + 1; k < n; k++) {
                if (std::abs(A.at(k, i)) > std::abs(A.at(max_row, i))) max_row = k;
            }
            
            // Swap
            for (int k = i; k < n; k++) std::swap(A.at(max_row, k), A.at(i, k));
            std::swap(b[max_row], b[i]);
            
            // Eliminate
            for (int k = i + 1; k < n; k++) {
                float factor = A.at(k, i) / A.at(i, i);
                b[k] -= factor * b[i];
                for (int j = i; j < n; j++) {
                    A.at(k, j) -= factor * A.at(i, j);
                }
            }
        }
        
        // Back substitute
        Vector x(n, 0.0f);
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (int j = i + 1; j < n; j++) sum += A.at(i, j) * x[j];
            x[i] = (b[i] - sum) / A.at(i, i);
        }
        
        return x;
    }
};

// Solve V * w = y using normal equations: (V^T V) w = V^T y
inline Vector least_squares(const std::vector<Vector>& V, const std::vector<float>& y, int n_dims) {
    int m = V.size();
    Matrix M(m, n_dims);
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n_dims; ++c) M.at(r, c) = V[r][c];
    }
    
    Matrix Mt = M.transpose();
    Matrix MtM = Mt.multiply(M); // n_dims x n_dims
    
    Vector Mt_y(n_dims, 0.0f);
    for (int r = 0; r < n_dims; ++r) {
        for (int c = 0; c < m; ++c) {
            Mt_y[r] += Mt.at(r, c) * y[c];
        }
    }
    
    try {
        return MtM.solve(Mt_y);
    } catch (...) {
        // If singular, return zero vector fallback
        return Vector(n_dims, 0.0f);
    }
}

class GroundingPipeline {
public:
    int n_dims;
    SOM* som;
    BindingMemory* binding;
    
    std::map<std::string, Vector> numeric_axes;
    std::map<std::string, Vector> numeric_decoders;
    std::map<std::string, Vector> category_centroids;
    
    Vector AXIS;
    
    GroundingPipeline(int nd, SOM* s = nullptr, BindingMemory* b = nullptr) 
        : n_dims(nd), som(s), binding(b) {
        AXIS = get_symbol_vector("__magnitude__", n_dims);
    }
    
    // Categorical
    void ground_categories(const std::vector<std::pair<Vector, std::string>>& labeled_data) {
        if (!som) return;
        std::map<std::string, std::vector<Vector>> acts;
        for (const auto& kv : labeled_data) {
            acts[kv.second].push_back(som->activation_map(kv.first));
        }
        
        for (const auto& kv : acts) {
            Vector centroid(som->n_neurons, 0.0f);
            for (const auto& a : kv.second) {
                for (size_t i = 0; i < a.size(); ++i) centroid[i] += a[i];
            }
            for (size_t i = 0; i < centroid.size(); ++i) centroid[i] /= kv.second.size();
            category_centroids[kv.first] = centroid;
        }
    }
    
    std::string recognize_category(const Vector& v) {
        if (!som || category_centroids.empty()) return "";
        Vector a = som->activation_map(v);
        
        std::string best_cat = "";
        float best_sim = -1.0f;
        for (const auto& kv : category_centroids) {
            float sim = cosine_similarity(a, kv.second);
            if (sim > best_sim) {
                best_sim = sim;
                best_cat = kv.first;
            }
        }
        return best_cat;
    }
    
    // Numeric
    void calibrate_numeric_sensors(const std::vector<std::string>& attributes, const std::vector<std::map<std::string, float>>& labeled_obs) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (const auto& a : attributes) {
            if (numeric_axes.count(a) == 0) {
                Vector v(n_dims);
                for (int i = 0; i < n_dims; ++i) v[i] = dist(rng);
                numeric_axes[a] = v; // Not normalized
            }
        }
        
        std::vector<Vector> V;
        for (const auto& obs : labeled_obs) {
            Vector v(n_dims, 0.0f);
            for (const auto& a : attributes) {
                if (obs.count(a)) {
                    for (int i = 0; i < n_dims; ++i) v[i] += obs.at(a) * numeric_axes[a][i];
                }
            }
            // Add noise
            for (int i = 0; i < n_dims; ++i) v[i] += 0.05f * dist(rng);
            V.push_back(v);
        }
        
        for (const auto& a : attributes) {
            std::vector<float> y;
            for (const auto& obs : labeled_obs) {
                y.push_back(obs.count(a) ? obs.at(a) : 0.0f);
            }
            numeric_decoders[a] = least_squares(V, y, n_dims);
        }
    }
    
    std::map<std::string, float> decode_numeric(const Vector& v, const std::vector<std::string>& attributes) {
        std::map<std::string, float> res;
        for (const auto& a : attributes) {
            if (numeric_decoders.count(a)) {
                res[a] = dot(v, numeric_decoders[a]);
            }
        }
        return res;
    }
    
    // Memory Binding
    void bind_perceived_quantities(const std::string& entity_name, const std::map<std::string, float>& decoded_vals) {
        if (!binding) return;
        Vector e_vec = get_symbol_vector(entity_name, n_dims);
        for (const auto& kv : decoded_vals) {
            Vector r_vec = get_symbol_vector(kv.first, n_dims);
            Vector o_vec = AXIS;
            for (auto& x : o_vec) x *= kv.second;
            binding->bind(e_vec, r_vec, o_vec);
        }
    }
    
    std::optional<float> query_bound_quantity(const std::string& entity_name, const std::string& relation_name, float conf_threshold = 0.9f) {
        if (!binding) return std::nullopt;
        Vector e_vec = get_symbol_vector(entity_name, n_dims);
        Vector r_vec = get_symbol_vector(relation_name, n_dims);
        
        auto res = binding->query(e_vec, r_vec);
        if (res.second >= conf_threshold) {
            return dot(res.first, AXIS);
        }
        return std::nullopt;
    }
};

// crispify_bridge.py
class BindingFactStore {
public:
    BindingMemory* bm;
    int n_dims;
    float conf_threshold;
    Vector AXIS;
    
    BindingFactStore(BindingMemory* m, int nd, float conf = 0.9f) 
        : bm(m), n_dims(nd), conf_threshold(conf) {
        AXIS = get_symbol_vector("__magnitude__", n_dims);
    }
    
    void bind(const std::string& entity, const std::string& rel, float value) {
        Vector o_vec = AXIS;
        for (auto& x : o_vec) x *= value;
        bm->bind(get_symbol_vector(entity, n_dims), get_symbol_vector(rel, n_dims), o_vec);
    }
    
    std::optional<float> fact(const std::string& entity, const std::string& rel) {
        auto res = bm->query(get_symbol_vector(entity, n_dims), get_symbol_vector(rel, n_dims));
        if (res.second < conf_threshold) return std::nullopt;
        return dot(res.first, AXIS);
    }
};

} // namespace grounding
} // namespace brain2
