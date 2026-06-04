#pragma once
/*
 * reasoning.hpp — Reasoning Engine, Component 12 of Brain v2
 *
 * System 2 thinking: slow, deliberate, step-by-step.
 * Works with Scratchpad + Symbolic to chain operations.
 *
 * Complements the fast pattern-matching brain (System 1).
 *
 * A ReasoningStep = {input_slot, op_symbol, arg_slot, output_slot}
 * Chain of steps = a reasoning program.
 *
 * Example: solve "a + b = ?"
 *   step 0: read("a"), apply("+"), read("b") → write("result")
 *
 * Example: proof chain "if A then B, if B then C → A implies C"
 *   step 0: read("A"), apply("->"), read("B")  → write("step1")
 *   step 1: read("step1"), apply("->"), read("C") → write("conclusion")
 *
 * Convergence: stop when output slot stops changing (delta < threshold)
 * or max_steps reached.
 *
 * Also supports:
 *   - Unary ops (negate, copy)
 *   - Conditional: only execute step if similarity > threshold
 *   - Loop: repeat a sub-chain until convergence
 */

#include "scratchpad.hpp"
#include "symbolic.hpp"
#include "predictor.hpp"

#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace brain2 {

struct ReasoningStep {
    std::string input_slot;   // scratchpad slot name for first arg
    std::string op_symbol;    // symbol from Symbolic table ("+", "->", etc.)
    std::string arg_slot;     // scratchpad slot for second arg (empty = unary)
    std::string output_slot;  // where to write result
    float       condition_sim;// min similarity of input to execute (0 = always)
    std::string condition_slot;// compare input_slot to this slot for condition
};

struct ReasoningResult {
    bool                            converged;    // did output stabilize?
    int                             steps_taken;
    float                           final_delta;  // last change magnitude
    std::vector<std::string>        trace;        // step descriptions
    std::vector<std::vector<float>> outputs;      // output vec at each step
};

class ReasoningEngine {
public:
    int   n_dims;
    int   max_steps;
    float convergence_threshold;

private:
    Symbolic*   symbolic_;   // non-owning
    Predictor*  predictor_;  // non-owning, optional — for neural step refinement
    std::unique_ptr<std::mutex> mtx_;

    static float cosine(const std::vector<float>& a,
                        const std::vector<float>& b) noexcept {
        float dot = 0.f, na = 0.f, nb = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    static float norm(const std::vector<float>& v) noexcept {
        float s = 0.f; for (auto x : v) s += x*x; return std::sqrt(s);
    }

public:
    ReasoningEngine() : n_dims(0), max_steps(20),
                        convergence_threshold(0.01f),
                        symbolic_(nullptr), predictor_(nullptr),
                        mtx_(std::make_unique<std::mutex>()) {}

    ReasoningEngine(Symbolic* sym, int n_dims,
                    int max_steps = 20,
                    float convergence_threshold = 0.01f,
                    Predictor* pred = nullptr)
        : n_dims(n_dims), max_steps(max_steps),
          convergence_threshold(convergence_threshold),
          symbolic_(sym), predictor_(pred),
          mtx_(std::make_unique<std::mutex>()) {}

    ReasoningEngine(ReasoningEngine&&)            = default;
    ReasoningEngine& operator=(ReasoningEngine&&) = default;
    ReasoningEngine(const ReasoningEngine&)       = delete;
    ReasoningEngine& operator=(const ReasoningEngine&) = delete;

    // Execute a single reasoning step on a scratchpad
    bool execute_step(const ReasoningStep& step, Scratchpad& pad) {
        auto input = pad.read(step.input_slot);
        if (norm(input) < 1e-8f) return false;  // no input

        // Check condition
        if (!step.condition_slot.empty() && step.condition_sim > 0.f) {
            auto cond_vec = pad.read(step.condition_slot);
            float sim = cosine(input, cond_vec);
            if (sim < step.condition_sim) return false;  // condition not met
        }

        std::vector<float> result;

        if (step.arg_slot.empty()) {
            // Unary op: apply(op, input, zeros)
            std::vector<float> zeros(n_dims, 0.f);
            result = symbolic_->apply(step.op_symbol, input, zeros);
        } else {
            auto arg = pad.read(step.arg_slot);
            result = symbolic_->apply(step.op_symbol, input, arg);
        }

        // Optional: refine result through predictor (neural correction)
        if (predictor_ && norm(result) > 1e-8f) {
            bool was_offline = predictor_->is_offline();
            predictor_->set_offline(true);
            auto refined = predictor_->step(result);
            // Blend: 70% symbolic + 30% neural (symbolic dominates)
            for (int i = 0; i < n_dims && i < (int)refined.size(); i++)
                result[i] = 0.7f * result[i] + 0.3f * refined[i];
            predictor_->set_offline(was_offline);
        }

        pad.write(step.output_slot, result, "result");
        return true;
    }

    // Execute a chain of reasoning steps
    ReasoningResult reason(const std::vector<ReasoningStep>& steps,
                           Scratchpad& pad) {
        std::lock_guard<std::mutex> lock(*mtx_);
        ReasoningResult result;
        result.converged   = false;
        result.steps_taken = 0;
        result.final_delta = 1.f;

        if (!symbolic_ || steps.empty()) return result;

        for (int iter = 0; iter < max_steps; iter++) {
            for (const auto& step : steps) {
                bool executed = execute_step(step, pad);
                if (executed) {
                    float delta = pad.delta(step.output_slot);
                    result.final_delta = delta;
                    result.steps_taken++;

                    // Record trace
                    std::string desc = step.input_slot + " " + step.op_symbol;
                    if (!step.arg_slot.empty()) desc += " " + step.arg_slot;
                    desc += " → " + step.output_slot;
                    result.trace.push_back(desc);
                    result.outputs.push_back(pad.read(step.output_slot));

                    if (delta < convergence_threshold) {
                        result.converged = true;
                        return result;
                    }
                }
            }
        }
        return result;
    }

    // Convenience: solve binary op "a OP b → result"
    // Writes a, b to pad, executes op, returns result vec
    std::vector<float> solve_binary(
            const std::string& op,
            const std::vector<float>& a,
            const std::vector<float>& b,
            Scratchpad& pad) {
        pad.write("_arg_a", a, "input");
        pad.write("_arg_b", b, "input");
        ReasoningStep step{"_arg_a", op, "_arg_b", "_result", 0.f, ""};
        execute_step(step, pad);
        return pad.read("_result");
    }

    // Inference chain: given premises, derive conclusion
    // premises: list of (slot_name, vec) to load into pad
    // chain: list of (op, input_slot, arg_slot, output_slot)
    std::vector<float> infer(
            const std::vector<std::pair<std::string, std::vector<float>>>& premises,
            const std::vector<ReasoningStep>& chain,
            Scratchpad& pad) {
        // Load premises
        for (const auto& [name, vec] : premises)
            pad.write(name, vec, "premise");

        auto res = reason(chain, pad);
        if (chain.empty()) return std::vector<float>(n_dims, 0.f);
        return pad.read(chain.back().output_slot);
    }

    // Loop: repeatedly apply a step until convergence or max_steps
    // Useful for: fixed-point computation, iterative refinement
    ReasoningResult loop_until_convergence(
            const ReasoningStep& step,
            Scratchpad& pad,
            int max_iters = -1) {
        std::lock_guard<std::mutex> lock(*mtx_);
        ReasoningResult result;
        result.converged   = false;
        result.steps_taken = 0;
        result.final_delta = 1.f;

        int limit = (max_iters > 0) ? max_iters : max_steps;
        for (int i = 0; i < limit; i++) {
            // Copy output to input for next iteration
            if (i > 0) pad.copy(step.output_slot, step.input_slot);

            bool executed = execute_step(step, pad);
            if (!executed) break;

            float delta = pad.delta(step.output_slot);
            result.steps_taken++;
            result.final_delta = delta;
            result.outputs.push_back(pad.read(step.output_slot));

            if (delta < convergence_threshold) {
                result.converged = true;
                break;
            }
        }
        return result;
    }

    bool has_symbolic() const noexcept { return symbolic_ != nullptr; }
    bool has_predictor() const noexcept { return predictor_ != nullptr; }

    void expand_dims(int new_dims) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (new_dims <= n_dims) return;
        n_dims = new_dims;
    }
};

} // namespace brain2
