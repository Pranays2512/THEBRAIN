#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include "crisp/engines/reasoning/learned_guidance.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace brain2 {
namespace reasoning {

// ── 8-Puzzle Domain ─────────────────────────────────────────────────────────
const std::vector<int> EIGHT_PUZZLE_GOAL = {1, 2, 3, 4, 5, 6, 7, 8, 0};

class EightPuzzle : public SearchProblem<std::vector<int>> {
private:
    std::vector<int> start_state;
    std::function<double(const std::vector<int>&)> hfn;
public:
    EightPuzzle(std::vector<int> start, std::function<double(const std::vector<int>&)> hfn = nullptr)
        : start_state(start), hfn(hfn) {}

    std::vector<int> initial() const override { return start_state; }
    
    bool is_goal(const std::vector<int>& s) const override { return s == EIGHT_PUZZLE_GOAL; }
    
    std::string key(const std::vector<int>& s) const {
        std::string k;
        for (int x : s) k += std::to_string(x) + ",";
        return k;
    }
    
    double heuristic(const std::vector<int>& s) const override {
        if (hfn) return hfn(s);
        return 0.0;
    }

    std::vector<std::tuple<std::string, std::vector<int>, double>> moves(const std::vector<int>& s) const override {
        std::vector<std::tuple<std::string, std::vector<int>, double>> res;
        int idx = 0;
        for (; idx < 9; idx++) if (s[idx] == 0) break;
        int r = idx / 3, c = idx % 3;
        
        const int dr[] = {-1, 1, 0, 0};
        const int dc[] = {0, 0, -1, 1};
        const char* names[] = {"up", "down", "left", "right"};
        
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr >= 0 && nr < 3 && nc >= 0 && nc < 3) {
                int nidx = nr * 3 + nc;
                auto ns = s;
                std::swap(ns[idx], ns[nidx]);
                res.push_back({"blank " + std::string(names[i]), ns, 1.0});
            }
        }
        return res;
    }
};

// ── Features & True Manhattan ───────────────────────────────────────────────
inline std::vector<double> eight_puzzle_features(const std::vector<int>& s) {
    std::vector<double> f(8, 0.0);
    for (int idx = 0; idx < 9; idx++) {
        int val = s[idx];
        if (val == 0) continue;
        int gr = (val - 1) / 3, gc = (val - 1) % 3;
        int r = idx / 3, c = idx % 3;
        f[val - 1] = std::abs(gr - r) + std::abs(gc - c);
    }
    return f;
}

inline double eight_puzzle_manhattan(const std::vector<int>& s) {
    auto f = eight_puzzle_features(s);
    double sum = 0;
    for (double x : f) sum += x;
    return sum;
}

inline std::vector<int> scramble_8puzzle(int depth, int seed = 0) {
    std::vector<int> s = EIGHT_PUZZLE_GOAL;
    
    for (int i = 0; i < depth; i++) {
        EightPuzzle p(s);
        auto m = p.moves(s);
        s = std::get<1>(m[(seed + i * 17) % m.size()]);
    }
    return s;
}

inline std::vector<std::pair<std::vector<int>, double>> collect_8puzzle_examples(
    int n_tasks = 80, int depth = 12, int seed = 0) 
{
    std::vector<std::pair<std::vector<int>, double>> out;
    for (int i = 0; i < n_tasks; i++) {
        auto start = scramble_8puzzle(depth, seed + i);
        EightPuzzle problem(start, eight_puzzle_manhattan);
        auto res = solve_astar(problem);
        if (!res.path.empty()) {
            std::vector<std::vector<int>> states = {start};
            for (const auto& step : res.path) states.push_back(step.second);
            int total = states.size() - 1;
            for (int j = 0; j < (int)states.size(); j++) {
                out.push_back({states[j], (double)(total - j)});
            }
        }
    }
    return out;
}

}}
