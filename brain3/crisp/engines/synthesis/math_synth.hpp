#pragma once
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <iostream>
#include <cmath>
#include "../reasoning/tree_reason.hpp"

namespace brain3 {
namespace engines {
namespace synthesis {

struct MathSynthState {
    std::vector<std::string> tokens;
    int depth;
    bool operator==(const MathSynthState& o) const {
        return depth == o.depth && tokens == o.tokens;
    }
};

}
}
}

namespace std {
template<> struct hash<brain3::engines::synthesis::MathSynthState> {
    size_t operator()(const brain3::engines::synthesis::MathSynthState& s) const {
        size_t h = s.depth;
        for (const auto& t : s.tokens) h ^= std::hash<std::string>{}(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
}

namespace brain3 {
namespace engines {
namespace synthesis {

using MathFunc = std::function<int(int, int)>;
using MathEnv = std::map<std::string, int>;
using MathComponent = std::pair<int, std::function<int(const MathEnv&, const std::vector<int>&)>>;

class MathSynthProblem : public brain2::reasoning::SearchProblem<MathSynthState> {
    std::vector<std::pair<std::pair<int, int>, int>> examples;
    std::string schema;
    std::vector<std::string> base_tokens;
    std::map<std::string, MathFunc> library;
    int max_len;
    std::map<std::string, MathComponent> comp;
    
public:
    MathSynthProblem(const std::vector<std::pair<std::pair<int, int>, int>>& ex, 
                     const std::string& sch, const std::vector<std::string>& base, 
                     const std::map<std::string, MathFunc>& lib, int ml = 6)
        : examples(ex), schema(sch), base_tokens(base), library(lib), max_len(ml) {
        
        comp["Z"] = {0, [](const MathEnv&, const std::vector<int>&){ return 0; }};
        comp["S"] = {1, [](const MathEnv&, const std::vector<int>& a){ return a[0] + 1; }};
        comp["P"] = {1, [](const MathEnv&, const std::vector<int>& a){ return std::max(0, a[0] - 1); }};
        
        for (auto v : {"a", "b", "r"}) {
            std::string vv = v;
            comp[vv] = {0, [vv](const MathEnv& e, const std::vector<int>&){ return e.at(vv); }};
        }
        for (const auto& [name, fn] : library) {
            auto lfn = fn;
            comp[name] = {2, [lfn](const MathEnv&, const std::vector<int>& a){ return lfn(a[0], a[1]); }};
        }
    }

    MathSynthState initial() const override { return {{}, 0}; }
    
    int eval_postfix(const std::vector<std::string>& tokens, const std::map<std::string, MathComponent>& c, const MathEnv& env) const {
        std::vector<int> stack;
        for (const auto& t : tokens) {
            auto it = c.find(t);
            if (it == c.end()) return -1;
            int arity = it->second.first;
            if (stack.size() < arity) return -1;
            std::vector<int> args(arity);
            for (int i = 0; i < arity; i++) {
                args[arity - 1 - i] = stack.back();
                stack.pop_back();
            }
            stack.push_back(it->second.second(env, args));
        }
        return stack.size() == 1 ? stack[0] : -1;
    }
    
    MathFunc make_function(const std::vector<std::string>& step_tokens) const {
        auto scomp = comp;
        std::map<std::string, MathComponent> bcomp;
        bcomp["Z"] = scomp["Z"];
        bcomp["S"] = scomp["S"];
        bcomp["P"] = scomp["P"];
        
        std::string survivor = (schema == "a") ? "b" : "a";
        bcomp[survivor] = {0, [survivor](const MathEnv& e, const std::vector<int>&){ return e.at(survivor); }};
        
        std::string sch = schema;
        auto bt = base_tokens;
        auto st = step_tokens;
        
        return [this, sch, bt, st, bcomp, scomp](int a, int b) {
            if (sch == "a") {
                int acc = eval_postfix(bt, bcomp, {{"b", b}});
                for (int i = 0; i < a; i++) {
                    acc = eval_postfix(st, scomp, {{"a", i}, {"b", b}, {"r", acc}});
                    if (acc < 0) return -1;
                }
                return acc;
            } else {
                int acc = eval_postfix(bt, bcomp, {{"a", a}});
                for (int j = 0; j < b; j++) {
                    acc = eval_postfix(st, scomp, {{"a", a}, {"b", j}, {"r", acc}});
                    if (acc < 0) return -1;
                }
                return acc;
            }
        };
    }

    bool is_goal(const MathSynthState& s) const override {
        if (s.depth != 1 || s.tokens.empty()) return false;
        auto f = make_function(s.tokens);
        for (const auto& ex : examples) {
            if (f(ex.first.first, ex.first.second) != ex.second) return false;
        }
        return true;
    }
    
    std::vector<std::tuple<std::string, MathSynthState, double>> moves(const MathSynthState& s) const override {
        std::vector<std::tuple<std::string, MathSynthState, double>> m;
        if (s.tokens.size() >= max_len) return m;
        
        for (const auto& [name, c] : comp) {
            if (s.depth >= c.first) {
                MathSynthState nxt = s;
                nxt.tokens.push_back(name);
                nxt.depth = s.depth - c.first + 1;
                m.push_back({name, nxt, 1.0});
            }
        }
        return m;
    }
};

class LearnedArithmetic {
public:
    std::map<std::string, MathFunc> lib;
    
    LearnedArithmetic() {
        // Just the stub for now, search can be invoked via MathSynthProblem + solve_astar
    }
};

} // namespace synthesis
} // namespace engines
} // namespace brain3
