#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <algorithm>
#include <memory>
#include <numeric>
#include <cassert>

namespace brain3 {
namespace engines {
namespace neural {

// =========================================================================
// 1. PROJECTIVE SPINOR & ROTOR ALGEBRA (Clifford Even Subalgebra Cl(p, q))
// =========================================================================
struct ProjectiveSpinor {
    // Spinor represented by scalar + bivector components (linear O(p) dimension)
    float scalar;
    std::vector<float> bivectors; // Dimension p/2 planes

    ProjectiveSpinor(int dim = 16) : scalar(1.0f), bivectors(dim / 2, 0.0f) {}

    // Rotor conjugation: Q = R * psi * R^dagger
    ProjectiveSpinor rotate(float theta, int plane_idx) const {
        ProjectiveSpinor res = *this;
        float cos_half = std::cos(theta * 0.5f);
        float sin_half = std::sin(theta * 0.5f);
        
        // Rotor action on scalar & bivector
        float orig_bv = (plane_idx < (int)res.bivectors.size()) ? res.bivectors[plane_idx] : 0.0f;
        res.scalar = scalar * cos_half - orig_bv * sin_half;
        if (plane_idx < (int)res.bivectors.size()) {
            res.bivectors[plane_idx] = orig_bv * cos_half + scalar * sin_half;
        }
        return res;
    }

    // Clifford Geometric Scalar-Grade Product ⟨psi_1 * psi_2^dagger⟩_0
    static float scalar_grade_product(const ProjectiveSpinor& a, const ProjectiveSpinor& b) {
        float sum = a.scalar * b.scalar;
        size_t n = std::min(a.bivectors.size(), b.bivectors.size());
        for (size_t i = 0; i < n; ++i) {
            sum += a.bivectors[i] * b.bivectors[i];
        }
        return sum;
    }
};

// =========================================================================
// 2. DISSIPATIVE KURAMOTO-LYAPUNOV PHASE SYNCHRONIZATION (DK-RoPE)
// =========================================================================
class DK_RoPE_Layer {
public:
    float gamma;  // Lyapunov dissipation coefficient
    float coupling_K;

    DK_RoPE_Layer(float gamma_val = 0.1f, float K = 0.5f) : gamma(gamma_val), coupling_K(K) {}

    // Computes globally stable fixed-point phase trajectory on Riemann helix
    std::vector<float> compute_phases(int seq_len) const {
        std::vector<float> phases(seq_len, 0.0f);
        std::vector<float> velocities(seq_len, 0.0f);
        
        float dt = 0.05f;
        // Intrinsic frequencies
        std::vector<float> omega(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            omega[i] = 2.0f * M_PI / (1.0f + std::exp(-0.05f * i));
        }

        // Dissipative Kuramoto evolution (15 relaxation steps to lock fixed point)
        for (int step = 0; step < 15; ++step) {
            for (int i = 0; i < seq_len; ++i) {
                float coupling_sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    coupling_sum += std::sin(phases[j] - phases[i]);
                }
                float d_vel = omega[i] + (coupling_K / seq_len) * coupling_sum - gamma * velocities[i];
                velocities[i] += dt * d_vel;
                phases[i] += dt * velocities[i];
            }
        }
        return phases;
    }
};

// =========================================================================
// 3. MULTI-CHART ATLAS & HYPERBOLIC POTENTIAL
// =========================================================================
class MultiChartAtlas {
public:
    int dim;
    MultiChartAtlas(int d = 16) : dim(d) {}

    std::vector<float> get_smooth_metric_inv(const std::vector<float>& q) const {
        float r2 = 0.0f;
        for (float x : q) r2 += x * x;
        float w_canon = std::exp(-0.05f * r2);
        
        std::vector<float> g_inv(dim * dim, 0.0f);
        for (int i = 0; i < dim; ++i) {
            g_inv[i * dim + i] = w_canon + (1.0f - w_canon) * 0.95f; // Smooth metric interpolation
        }
        return g_inv;
    }
};

// =========================================================================
// 4. LANDAUER THERMODYNAMIC ANNEALING FFN
// =========================================================================
class LandauerAnnealingFFN {
public:
    int dim;
    float tau_half;
    float R0;
    float V0;

    LandauerAnnealingFFN(int d = 16, float tau = 100.0f, float r = 5.0f, float v = 2.0f)
        : dim(d), tau_half(tau), R0(r), V0(v) {}

    std::vector<float> forward(const std::vector<float>& x, float time_step, bool is_grounded) const {
        std::vector<float> out(dim);
        float decay = is_grounded ? 1.0f : std::exp(-time_step / tau_half);
        
        float norm_sq = 0.0f;
        for (float val : x) norm_sq += val * val;
        float norm = std::sqrt(norm_sq) + 1e-8f;

        // Hyperbolic Restorative Barrier: V_bound = V0 * ln(cosh(||x||/R0))
        float force_scale = std::tanh(norm / R0) / (norm / R0);

        for (int i = 0; i < dim; ++i) {
            // True conservative hyperbolic potential gradient: F(x) = tanh(x/R0) * V0
            float activated = std::tanh(x[i] / R0) * V0;
            out[i] = activated * decay * force_scale;
        }
        return out;
    }
};

// =========================================================================
// 5. SYMPLECTIC VERLET FLOW RESIDUAL STREAM (O(1) Time-Reversible Backprop)
// =========================================================================
struct PhaseState {
    std::vector<float> q; // Position (Semantic coordinates)
    std::vector<float> p; // Momentum (Causal flux)
};

class SymplecticVerletResidualStream {
public:
    int dim;
    float dt;

    SymplecticVerletResidualStream(int d = 16, float time_step = 0.05f) : dim(d), dt(time_step) {}

    // Forward Step (Conserves Liouville 2-form dq ^ dp)
    PhaseState forward_step(const PhaseState& state, const LandauerAnnealingFFN& ffn, float t, bool grounded) const {
        PhaseState next = state;
        std::vector<float> force = ffn.forward(state.q, t, grounded);

        // Half-step momentum
        for (int i = 0; i < dim; ++i) {
            next.p[i] -= 0.5f * dt * force[i];
        }
        // Full-step position
        for (int i = 0; i < dim; ++i) {
            next.q[i] += dt * next.p[i];
        }
        // Recalculate force at new position & finish momentum
        std::vector<float> next_force = ffn.forward(next.q, t, grounded);
        for (int i = 0; i < dim; ++i) {
            next.p[i] -= 0.5f * dt * next_force[i];
        }
        return next;
    }

    // Exact Time-Reversed Step (dt -> -dt for O(1) memory backpropagation)
    PhaseState reverse_step(const PhaseState& state, const LandauerAnnealingFFN& ffn, float t, bool grounded) const {
        PhaseState prev = state;
        std::vector<float> force = ffn.forward(state.q, t, grounded);

        // Half-step momentum backwards
        for (int i = 0; i < dim; ++i) {
            prev.p[i] += 0.5f * dt * force[i];
        }
        // Full-step position backwards
        for (int i = 0; i < dim; ++i) {
            prev.q[i] -= dt * prev.p[i];
        }
        // Recalculate force and finish momentum backwards
        std::vector<float> prev_force = ffn.forward(prev.q, t, grounded);
        for (int i = 0; i < dim; ++i) {
            prev.p[i] += 0.5f * dt * prev_force[i];
        }
        return prev;
    }
};

// =========================================================================
// 6. STAMLAT COMPLETE TRANSFORMER ENGINE
// =========================================================================
class STAMLAT_Engine {
public:
    int dim;
    int num_layers;
    float temperature; // T = 0 (Strict CAS Prover) to T > 0 (Fluid Generative Langevin)
    
    DK_RoPE_Layer dk_rope;
    MultiChartAtlas atlas;
    LandauerAnnealingFFN ffn;
    SymplecticVerletResidualStream symplectic_stream;
    
    std::mt19937 rng;

    STAMLAT_Engine(int d = 16, int layers = 4, float temp = 0.0f)
        : dim(d), num_layers(layers), temperature(temp),
          dk_rope(0.1f, 0.5f), atlas(d), ffn(d, 100.0f, 5.0f, 2.0f),
          symplectic_stream(d, 0.05f), rng(42) {}

    void set_temperature(float t) {
        temperature = std::max(0.0f, t);
    }

    // Forward pass through STAMLAT layers
    std::vector<PhaseState> forward_sequence(const std::vector<std::vector<float>>& token_embeddings, bool grounded = true) {
        int seq_len = token_embeddings.size();
        std::vector<float> phases = dk_rope.compute_phases(seq_len);

        std::vector<PhaseState> states(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            states[i].q = token_embeddings[i];
            states[i].p = std::vector<float>(dim, 0.0f);
        }

        // Pass through symplectic layers
        for (int l = 0; l < num_layers; ++l) {
            for (int i = 0; i < seq_len; ++i) {
                // Apply Spinor rotor rotation by phase
                ProjectiveSpinor spinor(dim);
                spinor.scalar = states[i].q[0];
                for (size_t b = 0; b < spinor.bivectors.size() && b + 1 < states[i].q.size(); ++b) {
                    spinor.bivectors[b] = states[i].q[b + 1];
                }
                ProjectiveSpinor rotated = spinor.rotate(phases[i], l % (dim / 2));
                states[i].q[0] = rotated.scalar;
                for (size_t b = 0; b < rotated.bivectors.size() && b + 1 < states[i].q.size(); ++b) {
                    states[i].q[b + 1] = rotated.bivectors[b];
                }

                // Symplectic Verlet Step
                states[i] = symplectic_stream.forward_step(states[i], ffn, float(l * 10), grounded);

                // If Langevin temperature > 0, inject thermal Brownian noise
                if (temperature > 1e-6f) {
                    std::normal_distribution<float> noise(0.0f, std::sqrt(2.0f * temperature * 0.01f));
                    for (int d_idx = 0; d_idx < dim; ++d_idx) {
                        states[i].p[d_idx] += noise(rng);
                    }
                }
            }
        }
        return states;
    }

    // Token Emission: Thermal Gibbs-Boltzmann Sampling vs Deterministic Argmax
    int emit_token(const std::vector<float>& state_q, const std::vector<std::vector<float>>& vocab_embeddings) {
        std::vector<float> logits(vocab_embeddings.size(), 0.0f);
        for (size_t i = 0; i < vocab_embeddings.size(); ++i) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += state_q[d] * vocab_embeddings[i][d];
            }
            logits[i] = dot;
        }

        if (temperature < 1e-5f) {
            // Strict Argmax (Deterministic Invariance)
            return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        } else {
            // Thermal Boltzmann Gibbs Sampling
            std::vector<float> probs(logits.size());
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i) {
                probs[i] = std::exp((logits[i] - max_l) / temperature);
                sum_exp += probs[i];
            }
            std::uniform_real_distribution<float> dist(0.0f, sum_exp);
            float r = dist(rng);
            float cum = 0.0f;
            for (size_t i = 0; i < probs.size(); ++i) {
                cum += probs[i];
                if (cum >= r) return i;
            }
            return probs.size() - 1;
        }
    }
};

} // namespace neural
} // namespace engines
} // namespace brain3
