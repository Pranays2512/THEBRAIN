#pragma once
#include <vector>
#include <algorithm>
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

    // Compute error = actual - prediction; returns error vector reference.
    //
    // BOUNDS FIX: this loop used to run to n_dims unconditionally and index
    // actual[i] with no size check. Callers legitimately hand it shorter
    // vectors — Brain::daydream() feeds 128-dim imagination frames into
    // pc_som, which is sized som_rows*som_cols (256) — so the old loop read
    // past the end of the caller's vector. Dimensions beyond the supplied
    // input are treated as "unobserved" and contribute zero error, which is
    // the correct semantics for a partially-observed percept and keeps
    // error_norm comparable across input widths.
    const std::vector<float>& compute(const std::vector<float>& actual) {
        const int n = std::min(n_dims, (int)actual.size());
        float norm = 0.f;
        for (int i = 0; i < n; i++) {
            error[i] = actual[i] - prediction[i];
            norm += error[i] * error[i];
        }
        for (int i = n; i < n_dims; i++) error[i] = 0.f;   // unobserved ⇒ no error
        error_norm = std::sqrt(norm / (float)(n > 0 ? n : 1));
        return error;
    }

    // Returns error vector if surprise is high enough, else returns zeros.
    const std::vector<float>& propagate(const std::vector<float>& actual) {
        compute(actual);
        // We ALWAYS return the error for now to ensure learning happens at all scales
        return error;
    }

    // Update prediction toward actual (Hebbian)
    void update() {
        for (int i = 0; i < n_dims; i++)
            prediction[i] += lr * error[i];
    }

    bool should_propagate() const { return error_norm > threshold; }

    void expand_dims(int new_dims) {
        if (new_dims <= n_dims) return;
        prediction.resize(new_dims, 0.f);
        error.resize(new_dims, 0.f);
        n_dims = new_dims;
    }
};

} // namespace brain2
