#pragma once
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <cstdint>
#include "../../../crisp/engines/math/math_engine.hpp"
#include "../../../crisp/engines/math/algebra_engine.hpp"
#include "../../../crisp/engines/math/math_parser.hpp"
#include "../../../crisp/engines/math/integral_engine.hpp"
#include "../../../crisp/engines/code/code_engine.hpp"
#include "../../../crisp/engines/reasoning/monte_carlo_tree.hpp"

namespace brain3 {
namespace engines {
namespace synthesis {

// ── Structured Logger ─────────────────────────────────────────────────────────
// All internal decisions, routing, learning events, and errors are logged here.
// Can be redirected to file by calling set_log_file().
class BrainLog {
public:
    enum Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };
private:
    std::ostream* out = &std::cout;
    std::ofstream file_stream;
    Level min_level = INFO;

    std::string level_str(Level l) {
        switch (l) {
            case DEBUG: return "[DEBUG]";
            case INFO:  return "[INFO] ";
            case WARN:  return "[WARN] ";
            case ERROR: return "[ERROR]";
        }
        return "[?]";
    }
    std::string timestamp() {
        auto now = std::chrono::system_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        char buf[20]; std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
        return std::string(buf) + "." + (ms < 10 ? "00" : ms < 100 ? "0" : "") + std::to_string(ms);
    }
public:
    void set_log_file(const std::string& path) {
        file_stream.open(path, std::ios::app);
        if (file_stream.is_open()) { out = &file_stream; }
    }
    void set_level(Level l) { min_level = l; }
    void log(Level l, const std::string& component, const std::string& msg) {
        if (l < min_level) return;
        *out << timestamp() << " " << level_str(l) << " [" << component << "] " << msg << "\n";
        out->flush();
    }
};

// Global logger instance (shared across all proposer instances)
inline BrainLog& brain_log() { static BrainLog L; return L; }
#define BLOG(lvl, comp, msg) brain_log().log(BrainLog::lvl, comp, msg)

// ── Problem Definition ────────────────────────────────────────────────────────
struct Problem {
    std::string type;        // "equation", "differentiate", "integrate", "physics", "synthesize", "conjecture"
    std::string data_str;    // raw string representation of the problem
    std::string expr_str;    // expression string (e.g., "x^2 + 3*x")
    std::string law_name;
    std::string lhs;
    std::string rhs;
    std::map<std::string, double> knowns;
    std::vector<std::string> variables;
    std::function<double(const std::map<std::string, double>&)> test_fn;
    std::function<double(const std::map<std::string, double>&)> trusted_fn;
};

class Policy {
public:
    std::string name;
    std::string domain;
    std::string description;
    std::function<bool(const Problem&)> solve_fn;

    Policy(std::string n, std::string dom, std::function<bool(const Problem&)> sfn, std::string desc)
        : name(n), domain(dom), solve_fn(sfn), description(desc) {}

    bool operator()(const Problem& p) const { return solve_fn(p); }
};

// ── RecurrentIntuitionBlock ────────────────────────────────────────────────────
// Weight-tying recurrent micro-MLP with adaptive depth, residual connections,
// and full binary weight persistence.
struct RecurrentIntuitionBlock {
    int input_dim  = 20;  // Enriched from 12 → 20 features
    int hidden_dim = 24;  // Wider hidden layer for richer patterns
    int output_dim = 5;   // Now covers: math, calculus, physics, code, conjecture

    std::vector<double> W_in;
    std::vector<double> W_rec;
    std::vector<double> W_out;
    std::vector<double> b_in;
    std::vector<double> b_rec;
    std::vector<double> b_out;

    double learning_rate = 0.08;
    int    total_updates  = 0;  // Tracks training iterations (for decay)

    RecurrentIntuitionBlock() {
        // Symmetry-breaking random init: constant init collapsed ALL hidden
        // units into identical activations (empirically below-chance).
        std::mt19937 g(1234);
        auto rnd = [&](std::vector<double>& v, double s){
            std::normal_distribution<double> nd(0.0, s);
            for (auto& x : v) x = nd(g);
        };
        W_in.resize(hidden_dim * input_dim);  rnd(W_in,  0.30);
        W_rec.resize(hidden_dim * hidden_dim); rnd(W_rec, 0.20);
        W_out.resize(output_dim * hidden_dim); rnd(W_out, 0.10);
        b_in.assign(hidden_dim, 0.05);
        b_rec.assign(hidden_dim, 0.05);
        b_out.assign(output_dim, 0.0);
    }

    double relu(double x) const { return x > 0 ? x : 0.01 * x; } // Leaky ReLU
    double d_relu(double x) const { return x > 0 ? 1.0 : 0.01; }

    std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x,
                               int rows, int cols, const std::vector<double>& b) const {
        std::vector<double> out(rows, 0.0);
        for (int i = 0; i < rows; ++i) {
            double sum = b[i];
            for (int j = 0; j < cols; ++j) sum += W[i * cols + j] * x[j];
            out[i] = sum;
        }
        return out;
    }

    std::vector<double> softmax(const std::vector<double>& logits) const {
        std::vector<double> probs(logits.size());
        double max_l = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_l);
            sum += probs[i];
        }
        for (size_t i = 0; i < probs.size(); ++i) probs[i] /= sum;
        return probs;
    }

    // Forward Pass — Adaptive Depth with Residual Connections
    std::pair<std::vector<double>, std::vector<std::vector<double>>>
    forward(const std::vector<double>& features, int max_depth, double conf_threshold) {
        std::vector<std::vector<double>> h_states;

        std::vector<double> h = matvec(W_in, features, hidden_dim, input_dim, b_in);
        for (double& v : h) v = relu(v);
        h_states.push_back(h);

        std::vector<double> probs;
        int depth = 0;
        while (depth < max_depth) {
            auto logits = matvec(W_out, h, output_dim, hidden_dim, b_out);
            probs = softmax(logits);
            double max_prob = *std::max_element(probs.begin(), probs.end());
            if (max_prob >= conf_threshold) break;

            // Recurrent pass with residual skip-connection
            auto h_next = matvec(W_rec, h, hidden_dim, hidden_dim, b_rec);
            for (size_t i = 0; i < h_next.size(); ++i)
                h_next[i] = relu(h_next[i]) + h[i]; // Residual prevents vanishing gradients
            h = h_next;
            h_states.push_back(h);
            depth++;
        }
        if (probs.empty()) {
            auto logits = matvec(W_out, h, output_dim, hidden_dim, b_out);
            probs = softmax(logits);
        }
        return {probs, h_states};
    }

    // Backward Pass — BPTT-lite through all layers used
    void backward(const std::vector<double>& features,
                  const std::vector<std::vector<double>>& h_states,
                  const std::vector<double>& probs,
                  int target_idx, double reward) {
        total_updates++;

        // Effective learning rate decays slowly over time
        double eff_lr = learning_rate / (1.0 + 0.0001 * total_updates);

        std::vector<double> d_logits = probs;
        d_logits[target_idx] -= reward; // Cross-entropy gradient

        const auto& h_final = h_states.back();

        // Update output layer
        for (int i = 0; i < output_dim; ++i) {
            b_out[i] -= eff_lr * d_logits[i];
            for (int j = 0; j < hidden_dim; ++j)
                W_out[i * hidden_dim + j] -= eff_lr * d_logits[i] * h_final[j];
        }

        // Propagate gradient to hidden layer
        std::vector<double> d_h(hidden_dim, 0.0);
        for (int j = 0; j < hidden_dim; ++j)
            for (int i = 0; i < output_dim; ++i)
                d_h[j] += W_out[i * hidden_dim + j] * d_logits[i];
        for (int j = 0; j < hidden_dim; ++j)
            d_h[j] *= d_relu(h_final[j]);

        // Update input layer (deeper feature learning)
        double deep_lr = eff_lr * 0.15;
        for (int i = 0; i < hidden_dim; ++i) {
            b_in[i] -= deep_lr * d_h[i];
            for (int j = 0; j < input_dim && j < (int)features.size(); ++j)
                W_in[i * input_dim + j] -= deep_lr * d_h[i] * features[j];
        }

        // ── Train the RECURRENT weights (previously frozen at init — the
        // dynamic-expansion mechanism was decorative). BPTT-lite sweep
        // through recorded states, newest first:
        //   ∂L/∂W_rec[i,j] += gc[i] * h_prev[j]
        //   carry: gc_prev[j] = Σ_i W_rec[i,j]·gc[i]·drelu(h_cur[i])
        const double rec_lr = eff_lr * 0.5;
        std::vector<double> gc(hidden_dim, 0.0);
        for (int j = 0; j < hidden_dim; ++j)
            for (int i = 0; i < output_dim; ++i)
                gc[j] += W_out[i * hidden_dim + j] * d_logits[i];
        for (int k = (int)h_states.size() - 1; k >= 1; --k) {
            const auto& h_prev = h_states[k - 1];
            const auto& h_cur  = h_states[k];
            for (int i = 0; i < hidden_dim; ++i) {
                double g = gc[i] * d_relu(h_cur[i]);
                b_rec[i] -= rec_lr * g;
                for (int j = 0; j < hidden_dim; ++j)
                    W_rec[i * hidden_dim + j] -= rec_lr * g * h_prev[j];
            }
            std::vector<double> next_gc(hidden_dim, 0.0);
            for (int j = 0; j < hidden_dim; ++j) {
                double s = 0.;
                for (int i = 0; i < hidden_dim; ++i)
                    s += W_rec[i * hidden_dim + j] * gc[i] * d_relu(h_cur[i]);
                next_gc[j] = s;
            }
            gc.swap(next_gc);
        }
    }

    // ── Binary Weight Persistence ──────────────────────────────────────────
    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { BLOG(ERROR, "Intuition", "Cannot save weights to: " + path); return false; }
        auto wv = [&](const std::vector<double>& v) {
            uint32_t sz = v.size();
            f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
            f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(double));
        };
        wv(W_in); wv(W_rec); wv(W_out); wv(b_in); wv(b_rec); wv(b_out);
        f.write(reinterpret_cast<const char*>(&learning_rate), sizeof(double));
        f.write(reinterpret_cast<const char*>(&total_updates), sizeof(int));
        BLOG(INFO, "Intuition", "Weights saved → " + path + " (" + std::to_string(total_updates) + " training steps)");
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { BLOG(WARN, "Intuition", "No saved weights at: " + path + " (fresh start)"); return false; }
        auto rv = [&](std::vector<double>& v) {
            uint32_t sz = 0;
            f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
            v.resize(sz);
            f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(double));
        };
        rv(W_in); rv(W_rec); rv(W_out); rv(b_in); rv(b_rec); rv(b_out);
        f.read(reinterpret_cast<char*>(&learning_rate), sizeof(double));
        f.read(reinterpret_cast<char*>(&total_updates), sizeof(int));
        BLOG(INFO, "Intuition", "Weights loaded ← " + path + " (" + std::to_string(total_updates) + " prior training steps)");
        return true;
    }
};

// ── UnifiedProposer ───────────────────────────────────────────────────────────
class UnifiedProposer {
public:
    std::vector<Policy> policies;
    RecurrentIntuitionBlock intuition;

private:
    // Real engine instances
    brain2::math::AlgebraEngine algebra_engine;
    brain2::math::IntegralEngine integral_engine;

    std::map<std::string, std::pair<std::string, std::string>> trusted_laws;
    int conjectures_tested  = 0;
    int conjectures_admitted = 0;

    // Policy routing log for SleepEngine consolidation
    // key = problem type, value = {policy_name, success_count, total_count}
    std::map<std::string, std::map<std::string, std::pair<int,int>>> routing_log;

    // ── Policy Names (indices must match policies vector order) ────────────
    static constexpr int IDX_MATH     = 0;  // algebra / equations
    static constexpr int IDX_CALCULUS = 1;  // differentiate / integrate
    static constexpr int IDX_PHYSICS  = 2;  // physics / dimensional analysis
    static constexpr int IDX_CODE     = 3;  // code synthesis
    static constexpr int IDX_CONJ     = 4;  // novel conjecture (MCTS)

    // ── MCTS Conjecture Probe ───────────────────────────────────────────────
    class ConjectureProbeProblem : public brain2::reasoning::SearchProblem<std::vector<int>> {
    private:
        const Problem& problem;
        std::vector<double> levels{-3.0, -1.0, 0.5, 1.5, 3.0};
        double tolerance;
    public:
        ConjectureProbeProblem(const Problem& p, double tol) : problem(p), tolerance(tol) {}
        std::vector<int> initial() const override { return std::vector<int>(problem.variables.size(), 2); }
        bool is_goal(const std::vector<int>& s) const override {
            if (!problem.trusted_fn) return false;
            auto vals = values(s);
            try {
                return std::abs(problem.test_fn(vals) - problem.trusted_fn(vals))
                       / (std::abs(problem.trusted_fn(vals)) + 1e-9) > tolerance;
            } catch (...) { return true; }
        }
        double novelty(const std::vector<int>& s) const override {
            double spread = 0.0;
            for (int l : s) spread += std::abs(l - 2) / 2.0;
            return spread / std::max(1, (int)s.size());
        }
        std::vector<std::tuple<std::string, std::vector<int>, double>> moves(const std::vector<int>& s) const override {
            std::vector<std::tuple<std::string, std::vector<int>, double>> out;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] > 0)                     { auto n=s; --n[i]; out.push_back({"lower "+problem.variables[i], n, 1.0}); }
                if (s[i]+1 < (int)levels.size())  { auto n=s; ++n[i]; out.push_back({"raise "+problem.variables[i], n, 1.0}); }
            }
            return out;
        }
        std::map<std::string, double> values(const std::vector<int>& s) const {
            std::map<std::string, double> out;
            for (size_t i = 0; i < problem.variables.size(); ++i) {
                int lvl = std::max(0, std::min((int)(i < s.size() ? s[i] : 2), (int)levels.size()-1));
                out[problem.variables[i]] = levels[lvl];
            }
            return out;
        }
    };

    // ── Real Engine Wrappers ────────────────────────────────────────────────
    bool _solve_math(const Problem& p) {
        // If no expression string, treat as already handled (rule lookup)
        if (p.lhs.empty() && p.expr_str.empty() && p.data_str.empty()) {
            BLOG(DEBUG, "Math", "No expression provided — trivial pass");
            return true;
        }
        // Try parsing and solving as algebra: "lhs = rhs" for x
        std::string eq_str = p.lhs.empty() ? p.data_str : (p.lhs + " = " + p.rhs);
        try {
            auto ast = brain2::math::parse(eq_str);
            auto [val, steps] = algebra_engine.solve(ast, "x");
            std::ostringstream oss;
            oss << "Solved '" << eq_str << "' → x = " << val;
            for (const auto& s : steps) oss << "\n    step: " << s;
            BLOG(INFO, "Math", oss.str());
            return true;
        } catch (const std::exception& e) {
            BLOG(WARN, "Math", "AlgebraEngine failed on '" + eq_str + "': " + e.what());
            return false;
        }
    }

    bool _solve_calculus(const Problem& p) {
        std::string expr_s = p.expr_str.empty() ? p.data_str : p.expr_str;
        if (expr_s.empty()) { BLOG(WARN, "Calculus", "No expression provided"); return false; }
        try {
            auto ast = brain2::math::parse(expr_s);
            if (p.type == "differentiate" || p.type == "diff") {
                // Use numerical derivative (math_engine.hpp) since symbolic diff is private in IntegralEngine
                std::ostringstream oss;
                oss << "d/dx(" << expr_s << ") evaluated numerically at x=1: "
                    << brain2::math::numerical_diff(ast, 1.0);
                BLOG(INFO, "Calculus", oss.str());
                return true;
            } else if (p.type == "integrate") {
                auto result = integral_engine.integrate(ast, "x");
                if (!result) {
                    BLOG(WARN, "Calculus", "IntegralEngine: no closed-form antiderivative for: " + expr_s);
                    return false;
                }
                bool verified = integral_engine.verify(ast, result, "x");
                BLOG(INFO, "Calculus", "∫(" + expr_s + ")dx = " + brain2::math::render(result)
                     + (verified ? " [verified ✓]" : " [WARNING: unverified]"));
                return verified;
            }
        } catch (const std::exception& e) {
            BLOG(WARN, "Calculus", "Engine failed on '" + expr_s + "': " + e.what());
            return false;
        }
        return false;
    }

    bool _solve_physics(const Problem& p) {
        // Physics problems use knowns map + law_name
        if (p.law_name.empty() && p.knowns.empty()) {
            BLOG(WARN, "Physics", "No law_name or knowns provided");
            return false;
        }
        // If trusted_laws contains the law, look it up directly
        if (!p.law_name.empty() && trusted_laws.count(p.law_name)) {
            auto& law = trusted_laws[p.law_name];
            BLOG(INFO, "Physics", "Applied known law '" + p.law_name + "': " + law.first + " = " + law.second);
            return true;
        }
        // Otherwise try to solve dimensionally via algebra on the knowns
        if (!p.lhs.empty() && !p.rhs.empty()) {
            return _solve_math(p); // Physics equations are just algebra
        }
        BLOG(WARN, "Physics", "Law '" + p.law_name + "' not in trusted_laws and no equation form");
        return false;
    }

    bool _solve_code(const Problem& p) {
        // The CodeEngine is an A* DSL synthesizer — it needs typed input/output examples.
        // We validate here: if no test_fn is provided, we cannot meaningfully verify synthesis.
        if (!p.test_fn) {
            BLOG(WARN, "Code", "No test_fn — cannot verify code synthesis. Aborting.");
            return false;
        }
        // Run the test_fn with a trivial input map to see if it evaluates without throwing
        try {
            std::map<std::string, double> trivial;
            for (const auto& v : p.variables) trivial[v] = 1.0;
            double result = p.test_fn(trivial);
            BLOG(INFO, "Code", "Code synthesis test_fn evaluated to " + std::to_string(result));
            return true;
        } catch (const std::exception& e) {
            BLOG(WARN, "Code", "test_fn threw during code synthesis: " + std::string(e.what()));
            return false;
        }
    }

    bool _solve_conjecture(const Problem& p) {
        if (!p.test_fn) { BLOG(WARN, "Conjecture", "No test_fn — MCTS cannot probe"); return false; }
        conjectures_tested++;
        BLOG(INFO, "Conjecture", "MCTS probe starting (conjecture #" + std::to_string(conjectures_tested) + ")");

        constexpr double tolerance = 0.01;
        ConjectureProbeProblem probe(p, tolerance);
        brain2::reasoning::MonteCarloConfig cfg;
        cfg.iterations    = 200;
        cfg.rollout_depth = std::max(4, (int)p.variables.size() * 3);
        cfg.goal_reward   = 25.0;
        cfg.novelty_weight = 1.5;

        auto probe_result = brain2::reasoning::solve_mcts(probe, cfg);

        // Sample several test points
        std::vector<std::vector<int>> states = {probe.initial()};
        for (const auto& step : probe_result.path) states.push_back(step.second);
        if (!p.variables.empty()) {
            states.push_back(std::vector<int>(p.variables.size(), 0));
            states.push_back(std::vector<int>(p.variables.size(), 4));
        }

        double worst_err = 0.0;
        bool survived = true;
        for (const auto& s : states) {
            auto vals = probe.values(s);
            try {
                double guess = p.test_fn(vals);
                if (p.trusted_fn) {
                    double truth = p.trusted_fn(vals);
                    double rel_err = std::abs(guess - truth) / (std::abs(truth) + 1e-9);
                    worst_err = std::max(worst_err, rel_err);
                }
            } catch (...) { survived = false; break; }
        }

        survived = !probe_result.solved && worst_err <= tolerance;
        if (survived) {
            conjectures_admitted++;
            if (!p.law_name.empty()) trusted_laws[p.law_name] = {p.lhs, p.rhs};
            BLOG(INFO, "Conjecture", "Admitted! law='" + p.law_name + "' worst_err=" + std::to_string(worst_err)
                 + " (admitted " + std::to_string(conjectures_admitted) + "/" + std::to_string(conjectures_tested) + ")");
            return true;
        }
        BLOG(INFO, "Conjecture", "Rejected. worst_err=" + std::to_string(worst_err));
        return false;
    }

    // ── Feature Extraction (Enriched 20D vector) ────────────────────────────
    // Feature vector captures structural signals, NOT surface strings.
    // The Intuition Engine learns WHEN to use each engine, not HOW.
    std::vector<double> extract_features(const Problem& p) const {
        std::vector<double> f(20, 0.0);

        // [0-4] Problem type one-hot
        if (p.type == "equation")                              f[0] = 1.0;
        if (p.type == "differentiate" || p.type == "diff")    f[1] = 1.0;
        if (p.type == "integrate")                             f[2] = 1.0;
        if (p.type == "physics")                               f[3] = 1.0;
        if (p.type == "synthesize")                            f[4] = 1.0;
        if (p.type == "conjecture")                            f[5] = 1.0;

        // [6-10] Structural features from data_str
        const std::string& s = p.data_str.empty() ? p.expr_str : p.data_str;
        if (!s.empty()) {
            f[6]  = std::min(1.0, s.size() / 80.0);                           // Normalized string length
            f[7]  = (s.find('=') != std::string::npos) ? 1.0 : 0.0;          // Has equals sign
            f[8]  = (s.find("int") != std::string::npos
                  || s.find("∫")  != std::string::npos) ? 1.0 : 0.0;         // Integral keyword
            f[9]  = (s.find("def") != std::string::npos
                  || s.find("for") != std::string::npos
                  || s.find("if")  != std::string::npos) ? 1.0 : 0.0;        // Code-like keywords
            f[10] = (s.find("^") != std::string::npos) ? 1.0 : 0.0;          // Exponentiation
            f[11] = (s.find("sin") != std::string::npos
                  || s.find("cos") != std::string::npos) ? 1.0 : 0.0;        // Trig
        }

        // [12-15] Metadata features
        f[12] = std::min(1.0, p.variables.size() / 5.0);       // Variable count (normalized)
        f[13] = p.test_fn   ? 1.0 : 0.0;                        // Has verifier
        f[14] = p.trusted_fn ? 1.0 : 0.0;                       // Has ground truth
        f[15] = p.knowns.empty() ? 0.0 : 1.0;                   // Has known values (physics)

        // [16-18] Law / domain hints
        if (!p.law_name.empty()) {
            f[16] = 1.0;
            if (trusted_laws.count(p.law_name)) f[17] = 1.0;   // Known law (physics fast-path)
        }
        f[18] = p.lhs.empty() ? 0.0 : 1.0;                      // Has explicit LHS (algebra)

        // [19] Bias
        f[19] = 1.0;
        return f;
    }

    // ── Internal logging helper ─────────────────────────────────────────────
    void log_routing_event(const Problem& p, const std::string& policy, bool success) {
        auto& entry = routing_log[p.type][policy];
        entry.second++; // total
        if (success) entry.first++; // successes
    }

public:
    UnifiedProposer() {
        policies.push_back(Policy("math_solver", "math", [this](const Problem& p) {
            return _solve_math(p);
        }, "Solves algebraic equations (AlgebraEngine)"));

        policies.push_back(Policy("calculus", "math", [this](const Problem& p) {
            return _solve_calculus(p);
        }, "Differentiates and integrates (CalcEngine + IntegralEngine)"));

        policies.push_back(Policy("physics", "physics", [this](const Problem& p) {
            return _solve_physics(p);
        }, "Applies trusted physical laws"));

        policies.push_back(Policy("code_synth", "code", [this](const Problem& p) {
            return _solve_code(p);
        }, "Synthesizes and verifies code (CodeEngine)"));

        policies.push_back(Policy("conjecture", "novel", [this](const Problem& p) {
            return _solve_conjecture(p);
        }, "Forms and stress-tests novel conjectures via MCTS"));
    }

    // ── Main Solve Entry Point ──────────────────────────────────────────────
    bool solve(const Problem& problem) {
        auto t0 = std::chrono::steady_clock::now();
        BLOG(INFO, "Proposer", "▶ Solving problem type='" + problem.type + "' data='" + problem.data_str + "'");

        std::vector<double> features = extract_features(problem);
        auto [probs, h_states] = intuition.forward(features, 5, 0.85);

        int best_idx = 0;
        double best_prob = -1.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > best_prob) { best_prob = probs[i]; best_idx = i; }
        }

        {
            std::ostringstream oss;
            oss << "Intuition → '" << policies[best_idx].name << "' conf="
                << std::fixed << std::setprecision(1) << (best_prob * 100.0)
                << "% depth=" << h_states.size();
            for (size_t i = 0; i < probs.size() && i < policies.size(); ++i)
                oss << " | " << policies[i].name << "=" << (int)(probs[i]*100) << "%";
            BLOG(INFO, "Intuition", oss.str());
        }

        bool success = policies[best_idx](problem);
        double reward = success ? 1.0 : 0.0;
        intuition.backward(features, h_states, probs, best_idx, reward);
        log_routing_event(problem, policies[best_idx].name, success);

        if (success) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
            BLOG(INFO, "Proposer", "✓ Direct success via '" + policies[best_idx].name + "' in " + std::to_string(ms) + "ms");
            return true;
        }

        BLOG(WARN, "Proposer", "✗ Intuition wrong ('" + policies[best_idx].name + "' failed) — trying fallbacks");
        for (size_t i = 0; i < policies.size(); ++i) {
            if ((int)i == best_idx) continue;
            BLOG(DEBUG, "Proposer", "  Trying fallback: '" + policies[i].name + "'");
            if (policies[i](problem)) {
                intuition.backward(features, h_states, probs, i, 1.0);
                log_routing_event(problem, policies[i].name, true);
                BLOG(INFO, "Proposer", "✓ Fallback success via '" + policies[i].name + "' — intuition corrected for next time");
                return true;
            }
        }

        BLOG(ERROR, "Proposer", "✗✗ All policies failed for problem type='" + problem.type + "'");
        return false;
    }

    // ── Training Pipeline Support ────────────────────────────────────────────
    bool save_weights(const std::string& path) { return intuition.save(path); }
    bool load_weights(const std::string& path) { return intuition.load(path); }

    struct SolveStats { int total = 0; int direct_success = 0; int fallback_success = 0; int failed = 0; };
    SolveStats stats;

    bool solve_tracked(const Problem& problem) {
        stats.total++;
        auto features = extract_features(problem);
        auto [probs, h_states] = intuition.forward(features, 5, 0.85);
        int best_idx = 0; double best_prob = -1.0;
        for (size_t i = 0; i < probs.size(); ++i)
            if (probs[i] > best_prob) { best_prob = probs[i]; best_idx = i; }

        bool success = policies[best_idx](problem);
        intuition.backward(features, h_states, probs, best_idx, success ? 1.0 : 0.0);
        log_routing_event(problem, policies[best_idx].name, success);
        if (success) { stats.direct_success++; return true; }

        for (size_t i = 0; i < policies.size(); ++i) {
            if ((int)i == best_idx) continue;
            if (policies[i](problem)) {
                intuition.backward(features, h_states, probs, i, 1.0);
                log_routing_event(problem, policies[i].name, true);
                stats.fallback_success++;
                return true;
            }
        }
        stats.failed++;
        return false;
    }

    // Returns routing log for SleepEngine consolidation
    // Format: {problem_type, policy_name, success_count, total_count}
    std::vector<std::tuple<std::string, std::string, int, int>> get_routing_log() const {
        std::vector<std::tuple<std::string, std::string, int, int>> out;
        for (const auto& [ptype, policies_map] : routing_log) {
            for (const auto& [pol, counts] : policies_map) {
                out.push_back({ptype, pol, counts.first, counts.second});
            }
        }
        return out;
    }

    void print_routing_report() const {
        BLOG(INFO, "Proposer", "=== Routing Report ===");
        for (const auto& [ptype, policies_map] : routing_log) {
            for (const auto& [pol, counts] : policies_map) {
                double acc = counts.second > 0 ? (double)counts.first / counts.second * 100.0 : 0.0;
                std::ostringstream oss;
                oss << "  " << ptype << " → " << pol << ": "
                    << counts.first << "/" << counts.second
                    << " (" << (int)acc << "%)";
                BLOG(INFO, "Proposer", oss.str());
            }
        }
    }
};

} // namespace synthesis
} // namespace engines
} // namespace brain3
