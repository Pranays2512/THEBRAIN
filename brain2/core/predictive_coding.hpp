#pragma once
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace brain2 {

struct PredictiveCodingLayer {
    int   n_dims    = 0;
    float threshold = 0.05f;
    float lr        = 0.01f;

    std::vector<float> prediction;
    std::vector<float> error;
    float              error_norm = 0.f;

    PredictiveCodingLayer() = default;
    PredictiveCodingLayer(int n, float thr = 0.05f, float lr_ = 0.01f)
        : n_dims(n), threshold(thr), lr(lr_),
          prediction(n, 0.f), error(n, 0.f) {}

    // Compute error = actual - prediction; returns error vector reference
    const std::vector<float>& compute(const std::vector<float>& actual) {
        float norm = 0.f;
        for (int i = 0; i < n_dims; i++) {
            error[i] = actual[i] - prediction[i];
            norm += error[i] * error[i];
        }
        error_norm = std::sqrt(norm / (float)(n_dims > 0 ? n_dims : 1));
        return error;
    }

    // Returns error vector if surprise is high enough, else returns zeros.
    const std::vector<float>& propagate(const std::vector<float>& actual) {
        compute(actual);
        if (!should_propagate()) {
            std::fill(error.begin(), error.end(), 0.f);
            error_norm = 0.f;
        }
        return error;
    }

    // Update prediction toward actual (Hebbian)
    void update() {
        for (int i = 0; i < n_dims; i++)
            prediction[i] += lr * error[i];
    }

    bool should_propagate() const { return error_norm > threshold; }
};

} // namespace brain2
