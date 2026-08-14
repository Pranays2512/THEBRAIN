#pragma once
#include <string>
#include <vector>
#include <set>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include "crisp/engines/reasoning/tree_reason.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"

namespace brain2 {
namespace reasoning {

class PlanningError : public std::runtime_error {
public:
    PlanningError(const std::string& msg) : std::runtime_error(msg) {}
};

struct PlanStep {
    std::string action;
    std::set<std::string> uses;
    std::set<std::string> makes;
};

struct Plan {
    bool found;
    std::vector<PlanStep> steps;
    std::string goal;

    std::string explain() const {
        if (!found) return "  no plan reaches " + goal + " from what is known";
        std::string res;
        for (size_t i = 0; i < steps.size(); i++) {
            res += "  " + std::to_string(i + 1) + ". " + steps[i].action + ": use [";
            bool first = true;
            for (const auto& u : steps[i].uses) { if (!first) res += ", "; res += u; first = false; }
            if (steps[i].uses.empty()) res += "-";
            res += "] -> get [";
            first = true;
            for (const auto& m : steps[i].makes) { if (!first) res += ", "; res += m; first = false; }
            res += "]\n";
        }
        return res;
    }
};

class PlanningEngine;

class CraftPlan : public SearchProblem<std::set<std::string>> {
private:
    PlanningEngine* e;
    std::set<std::string> have;
    std::string goal;
public:
    CraftPlan(PlanningEngine* e, std::set<std::string> have, std::string goal) : e(e), have(have), goal(goal) {}

    std::set<std::string> initial() const override { return have; }

    bool is_goal(const std::set<std::string>& state) const override { return state.count(goal) > 0; }

    std::string key(const std::set<std::string>& state) const {
        std::string k;
        for (const auto& s : state) k += s + ",";
        return k;
    }

    double heuristic(const std::set<std::string>& state) const override {
        return state.count(goal) > 0 ? 0.0 : 1.0;
    }

    std::vector<std::tuple<std::string, std::set<std::string>, double>> moves(const std::set<std::string>& state) const override;
};

class PlanningEngine {
public:
    ReasoningEngine kb;

    std::set<std::string> actions() const {
        std::set<std::string> res;
        for (const auto& f : kb.facts) {
            if (f.rel == "requires" || f.rel == "produces") res.insert(f.subj);
        }
        return res;
    }

    std::set<std::string> requires_for(const std::string& action) const {
        std::set<std::string> res;
        for (const auto& f : kb.facts) if (f.subj == action && f.rel == "requires") res.insert(f.obj);
        return res;
    }

    std::set<std::string> produces_for(const std::string& action) const {
        std::set<std::string> res;
        for (const auto& f : kb.facts) if (f.subj == action && f.rel == "produces") res.insert(f.obj);
        return res;
    }

    void define_action(const std::string& name, const std::vector<std::string>& reqs, const std::vector<std::string>& prods) {
        if (prods.empty()) throw PlanningError("action " + name + " must produce something");
        for (const auto& r : reqs) kb.learn(name, "requires", r);
        for (const auto& p : prods) kb.learn(name, "produces", p);
    }

    Plan plan(const std::set<std::string>& have, const std::string& goal, int max_nodes = 200000) {
        CraftPlan prob(this, have, goal);
        auto res = solve_astar(prob, max_nodes);
        if (res.path.empty()) return {false, {}, goal};
        
        std::vector<PlanStep> steps;
        for (const auto& step : res.path) {
            steps.push_back({step.first, requires_for(step.first), produces_for(step.first)});
        }
        return {true, steps, goal};
    }
};

inline std::vector<std::tuple<std::string, std::set<std::string>, double>> CraftPlan::moves(const std::set<std::string>& state) const {
    std::vector<std::tuple<std::string, std::set<std::string>, double>> res;
    for (const auto& a : e->actions()) {
        auto req = e->requires_for(a);
        auto prod = e->produces_for(a);
        bool has_all_reqs = true;
        for (const auto& r : req) if (!state.count(r)) has_all_reqs = false;
        
        bool gives_new = false;
        for (const auto& p : prod) if (!state.count(p)) gives_new = true;
        
        if (has_all_reqs && gives_new) {
            auto ns = state;
            for (const auto& p : prod) ns.insert(p);
            res.push_back({a, ns, 1.0});
        }
    }
    return res;
}

}}
