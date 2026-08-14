/**
 * brain3/core/algorithmic_policy_engine.hpp
 *
 * THE BRAIN 3: Formal Algorithmic Policy & Invariant Engine
 *
 * Role:
 *   Stores deterministic mathematical invariants, state transitions, complexity budgets,
 *   and algorithmic policies. Serves as the "Mind" that instructs the "LLM Mouth" on
 *   how to synthesize code without hardcoded templates.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

namespace brain3 {
namespace core {

struct AlgorithmicPolicy {
    std::string problem_id;
    std::string paradigm;
    std::string mathematical_invariant;
    std::string transition_recurrence;
    std::string time_complexity_budget;
    std::string space_complexity_budget;
    std::string io_policy;
    std::string gc_safety_policy;
    std::vector<std::string> constraints;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"problem_id\": \"" << problem_id << "\",\n";
        oss << "  \"paradigm\": \"" << paradigm << "\",\n";
        oss << "  \"mathematical_invariant\": \"" << mathematical_invariant << "\",\n";
        oss << "  \"transition_recurrence\": \"" << transition_recurrence << "\",\n";
        oss << "  \"time_complexity_budget\": \"" << time_complexity_budget << "\",\n";
        oss << "  \"space_complexity_budget\": \"" << space_complexity_budget << "\",\n";
        oss << "  \"io_policy\": \"" << io_policy << "\",\n";
        oss << "  \"gc_safety_policy\": \"" << gc_safety_policy << "\"\n";
        oss << "}";
        return oss.str();
    }

    std::string to_mouth_prompt(const std::string& target_language = "Java") const {
        std::ostringstream oss;
        oss << "🧠 [THE BRAIN ALGORITHMIC POLICY SPECIFICATION]\n";
        oss << "Synthesize a production-grade, highly optimized competitive programming solution in " << target_language << ".\n\n";
        oss << "• Target Language: " << target_language << "\n";
        oss << "• Algorithmic Paradigm: " << paradigm << "\n";
        oss << "• Core Invariant / Mathematical Theorem: " << mathematical_invariant << "\n";
        if (!transition_recurrence.empty()) {
            oss << "• Recurrence / State Transitions: " << transition_recurrence << "\n";
        }
        oss << "• Asymptotic Time Budget: " << time_complexity_budget << "\n";
        oss << "• Space Complexity Budget: " << space_complexity_budget << "\n";
        oss << "• I/O Constraint: " << io_policy << "\n";
        oss << "• Memory Safety Rule: " << gc_safety_policy << "\n\n";
        oss << "The Brain will audit and verify your code in an isolated sandbox with javac and execution telemetry.\n";
        return oss.str();
    }
};

class AlgorithmicPolicyEngine {
private:
    std::unordered_map<std::string, AlgorithmicPolicy> policies_;

public:
    AlgorithmicPolicyEngine() {
        _init_canonical_policies();
    }

    void register_policy(const AlgorithmicPolicy& policy) {
        policies_[policy.problem_id] = policy;
    }

    bool has_policy(const std::string& key) const {
        return policies_.find(key) != policies_.end();
    }

    AlgorithmicPolicy get_policy(const std::string& key) const {
        auto it = policies_.find(key);
        if (it != policies_.end()) return it->second;
        return AlgorithmicPolicy{};
    }

    std::vector<std::string> list_policies() const {
        std::vector<std::string> res;
        for (const auto& kv : policies_) res.push_back(kv.first);
        return res;
    }

private:
    void _init_canonical_policies() {
        // Policy 1: Monge / Quadrangle Inequality D&C DP Optimization (CF 868F)
        register_policy({
            "divide_and_conquer_dp_monge",
            "Divide and Conquer DP Optimization",
            "Quadrangle Inequality: C(a, c) + C(b, d) <= C(a, d) + C(b, c) for a <= b <= c <= d implies monotonic optimal split opt(i, j) <= opt(i, j+1)",
            "DP[k][mid] = min_{optL <= p <= min(mid, optR)} (DP[k-1][p-1] + Cost(p, mid)) with 2-pointer frequency tracking",
            "O(K * N log N)",
            "O(N) with primitive 1D arrays",
            "Custom FastScanner byte-level buffer; PrintWriter buffered output",
            "Zero heap allocations inside recursion; preallocated flat 1D tables to prevent JVM GC pauses",
            {"N <= 100,000", "K <= 20", "Time Limit: 1.0s"}
        });

        // Policy 2: Offline Segment Tree over Preceding Occurrences (CF 1000F)
        register_policy({
            "offline_segment_tree_frequency",
            "Offline Segment Tree with Minimum Queries",
            "Element a[i] has frequency 1 in [L, R] iff prev[i] < L and next[i] > R; min_pos in [L, R] with prev[pos] < L",
            "Store prev[i] in segment tree indexed by i; on step R, deactivate prev[R] and activate R with value prev[R]",
            "O((N + Q) log N)",
            "O(N + Q) with primitive arrays",
            "Custom FastScanner with 64KB byte buffer",
            "Zero boxed Integer allocations; use primitive int[] for tree nodes and queries",
            {"N <= 500,000", "Q <= 500,000", "Time Limit: 1.5s"}
        });

        // Policy 3: Eulerian Bipartite Graph Decomposition (CF 547D)
        register_policy({
            "eulerian_bipartite_graph_coloring",
            "Eulerian Circuit Decomposition on Coordinate Bipartite Graph",
            "Form bipartite graph between X and Y coordinates. Pair odd-degree vertices with dummy edges to make all degrees even. Eulerian circuit alternate edge coloring achieves |deg_r - deg_b| <= 1.",
            "Hierholzer algorithm with head/next linked array representation; alternate color on edge traversal",
            "O(N + MAX_COORD)",
            "O(N + MAX_COORD)",
            "Custom FastScanner or BufferedReader with StringTokenizer",
            "Flattened forward-star edge arrays (head[], to[], next[], used[]) with zero Edge object instantiations",
            {"N <= 200,000", "X_i, Y_i <= 200,000", "Time Limit: 2.0s"}
        });

        // Policy 4: Multiplicative Expectation Number Theory DP (CF 1097D)
        register_policy({
            "multiplicative_expectation_number_theory",
            "Prime Factorization & Multiplicative Expectation DP",
            "E[N] = prod E[p_i^{a_i}] due to multiplicative independence of uniform random divisor steps over prime factorizations",
            "DP[step][v] = sum_{u=v}^{a_i} (DP[step-1][u] * inv[u+1]) mod (10^9 + 7)",
            "O(omega(N) * log^2(N) * K)",
            "O(log N)",
            "Fast I/O with standard stream",
            "Modular inverse precomputation table up to max exponent (<= 50)",
            {"N <= 10^12", "K <= 10,000", "Time Limit: 1.0s"}
        });
    }
};

} // namespace core
} // namespace brain3
