#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include <iostream>

namespace brain3 {
namespace engines {
namespace synthesis {

// ── Policy: a rule that derives target from inputs ────────────────────────────
struct Policy {
    std::string target;
    std::vector<std::string> inputs;
    // expr stored as a lambda; for display use expr_str
    std::string expr_str;
    std::function<double(const std::map<std::string, double>&)> eval;
};

// ── MultiPolicyMemory: many policies per target ───────────────────────────────
class MultiPolicyMemory {
private:
    std::map<std::string, std::vector<Policy>> by_target;

public:
    void add(const Policy& p) {
        by_target[p.target].push_back(p);
    }

    const std::vector<Policy>& candidates(const std::string& target) const {
        static const std::vector<Policy> empty;
        auto it = by_target.find(target);
        return it != by_target.end() ? it->second : empty;
    }

    bool contains(const std::string& target) const {
        return by_target.count(target) > 0;
    }
};

// ── Groundability: optimistic [0,1] estimate of whether rel can be resolved ───
double groundable(const std::string& rel,
                  const MultiPolicyMemory& mem,
                  std::function<bool(const std::string&)> kb,
                  std::vector<std::string> seen = {});

double groundable(const std::string& rel,
                  const MultiPolicyMemory& mem,
                  std::function<bool(const std::string&)> kb,
                  std::vector<std::string> seen)
{
    // Cycle guard
    if (std::find(seen.begin(), seen.end(), rel) != seen.end()) return 0.0;
    if (kb(rel)) return 1.0;  // fact present

    const auto& cands = mem.candidates(rel);
    if (cands.empty()) return 0.0;  // leaf, not a fact

    double best = 0.0;
    seen.push_back(rel);
    for (const auto& p : cands) {
        double sum = 0.0;
        for (const auto& inp : p.inputs)
            sum += groundable(inp, mem, kb, seen);
        double score = p.inputs.empty() ? 1.0 : sum / p.inputs.size();
        best = std::max(best, score);
    }
    return best;
}

double proposer_score(const Policy& policy,
                      const MultiPolicyMemory& mem,
                      std::function<bool(const std::string&)> kb)
{
    if (policy.inputs.empty()) return 1.0;
    double sum = 0.0;
    for (const auto& inp : policy.inputs)
        sum += groundable(inp, mem, kb);
    return sum / policy.inputs.size();
}

// ── Solver: goal-directed policy chaining with optional proposer ordering ─────
class PolicyProposerSolver {
private:
    std::map<std::string, double> facts;
    const MultiPolicyMemory& mem;
    bool use_proposer;

public:
    int work = 0;
    std::map<std::string, double> solved;

    PolicyProposerSolver(const std::map<std::string, double>& kb,
                         const MultiPolicyMemory& m,
                         bool proposer)
        : facts(kb), mem(m), use_proposer(proposer) {}

    bool has_fact(const std::string& rel) const {
        return facts.count(rel) > 0;
    }

    double* solve(const std::string& rel, std::vector<std::string> seen = {}) {
        work++;
        if (std::find(seen.begin(), seen.end(), rel) != seen.end()) return nullptr;
        if (solved.count(rel)) return &solved[rel];
        if (facts.count(rel)) { solved[rel] = facts.at(rel); return &solved[rel]; }

        auto cands = mem.candidates(rel);
        if (use_proposer) {
            std::sort(cands.begin(), cands.end(), [&](const Policy& a, const Policy& b) {
                auto kb = [&](const std::string& r){ return has_fact(r); };
                return proposer_score(a, mem, kb) > proposer_score(b, mem, kb);
            });
        }

        seen.push_back(rel);
        for (const auto& p : cands) {
            std::map<std::string, double> env;
            bool ok = true;
            for (const auto& inp : p.inputs) {
                double* v = solve(inp, seen);
                if (!v) { ok = false; break; }
                env[inp] = *v;
            }
            if (ok && p.eval) {
                solved[rel] = p.eval(env);
                return &solved[rel];
            }
        }
        return nullptr;
    }
};

}}}
