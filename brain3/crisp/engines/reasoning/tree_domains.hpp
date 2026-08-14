#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <tuple>
#include <map>

namespace brain2 {
namespace reasoning {

// ── N-Queens ──────────────────────────────────────────────────────────────
class NQueens : public SearchProblem<std::vector<int>> {
private:
    int n;
public:
    NQueens(int n) : n(n) {}

    std::vector<int> initial() const override {
        return {};
    }

    bool is_goal(const std::vector<int>& s) const override {
        return s.size() == n;
    }

    std::string key(const std::vector<int>& s) const {
        std::string k;
        for (int x : s) k += std::to_string(x) + ",";
        return k;
    }

    double heuristic(const std::vector<int>& s) const override {
        return n - s.size();
    }

    std::vector<std::tuple<std::string, std::vector<int>, double>> moves(const std::vector<int>& s) const override {
        std::vector<std::tuple<std::string, std::vector<int>, double>> res;
        int row = s.size();
        for (int col = 0; col < n; col++) {
            bool ok = true;
            for (size_t r = 0; r < s.size(); r++) {
                int c = s[r];
                if (col == c || std::abs(col - c) == std::abs(row - (int)r)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                auto ns = s;
                ns.push_back(col);
                res.push_back({"queen at (row " + std::to_string(row) + ", col " + std::to_string(col) + ")", ns, 1.0});
            }
        }
        return res;
    }
};

// ── Water Jugs ──────────────────────────────────────────────────────────────
class WaterJugs : public SearchProblem<std::vector<int>> {
private:
    std::vector<int> caps;
    int target;
public:
    WaterJugs(std::vector<int> caps, int target) : caps(caps), target(target) {}

    std::vector<int> initial() const override {
        return std::vector<int>(caps.size(), 0);
    }

    bool is_goal(const std::vector<int>& s) const override {
        for (int x : s) if (x == target) return true;
        return false;
    }

    std::string key(const std::vector<int>& s) const {
        std::string k;
        for (int x : s) k += std::to_string(x) + ",";
        return k;
    }

    std::vector<std::tuple<std::string, std::vector<int>, double>> moves(const std::vector<int>& s) const override {
        std::vector<std::tuple<std::string, std::vector<int>, double>> res;
        for (size_t i = 0; i < caps.size(); i++) {
            if (s[i] < caps[i]) {
                auto ns = s; ns[i] = caps[i];
                res.push_back({"fill jug" + std::to_string(i) + " (->" + std::to_string(caps[i]) + "L)", ns, 1.0});
            }
            if (s[i] > 0) {
                auto ns = s; ns[i] = 0;
                res.push_back({"empty jug" + std::to_string(i), ns, 1.0});
            }
            for (size_t j = 0; j < caps.size(); j++) {
                if (i != j && s[i] > 0 && s[j] < caps[j]) {
                    int amt = std::min(s[i], caps[j] - s[j]);
                    auto ns = s; ns[i] -= amt; ns[j] += amt;
                    res.push_back({"pour jug" + std::to_string(i) + "->jug" + std::to_string(j), ns, 1.0});
                }
            }
        }
        return res;
    }
};

// ── Rewrite / Proof ─────────────────────────────────────────────────────────
class Rewrite : public SearchProblem<std::string> {
private:
    std::string start, goal;
    std::vector<std::pair<std::string, std::string>> rules;
public:
    Rewrite(std::string start, std::string goal, std::vector<std::pair<std::string, std::string>> rules) 
        : start(start), goal(goal), rules(rules) {}

    std::string initial() const override { return start; }

    bool is_goal(const std::string& s) const override { return s == goal; }

    std::string key(const std::string& s) const { return s; }

    double heuristic(const std::string& s) const override { return (s == goal) ? 0.0 : 1.0; }

    std::vector<std::tuple<std::string, std::string, double>> moves(const std::string& s) const override {
        std::vector<std::tuple<std::string, std::string, double>> res;
        for (const auto& rule : rules) {
            size_t i = s.find(rule.first);
            while (i != std::string::npos) {
                std::string ns = s.substr(0, i) + rule.second + s.substr(i + rule.first.size());
                res.push_back({"apply  " + rule.first + " -> " + rule.second, ns, 1.0});
                i = s.find(rule.first, i + 1);
            }
        }
        return res;
    }
};

}}
