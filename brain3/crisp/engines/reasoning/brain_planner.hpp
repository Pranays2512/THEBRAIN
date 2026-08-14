#pragma once
#include <string>
#include <vector>
#include <set>
#include <memory>
#include <iostream>
#include <algorithm>

#include "fuzzy/core/brain.hpp"
#include "crisp/engines/reasoning/tree_reason.hpp"

namespace brain2 {
namespace reasoning {

class FactWorld {
private:
    std::shared_ptr<brain2::Brain> b;
    std::map<std::string, std::vector<float>> vecs;
    std::set<std::string> _actions;

    std::vector<float> vec(const std::string& token) {
        if (!vecs.count(token)) {
            size_t h = std::hash<std::string>{}(token) % 4294967296ULL;
            std::mt19937 gen(h);
            std::normal_distribution<float> d(0, 1);
            std::vector<float> v(64);
            for (auto& val : v) val = d(gen);
            vecs[token] = v;
        }
        return vecs[token];
    }

    std::optional<std::string> _decode(const std::vector<float>& v) {
        float norm = 0;
        for (float x : v) norm += x * x;
        if (norm < 1e-8) return std::nullopt;
        
        std::string best_t;
        float best_s = -1e9;
        for (const auto& [t, tv] : vecs) {
            float s = 0;
            float tnorm = 0;
            for (size_t i = 0; i < 64; i++) {
                s += (v[i] / std::sqrt(norm)) * tv[i];
                tnorm += tv[i] * tv[i];
            }
            s /= std::sqrt(tnorm + 1e-8);
            if (s > best_s) { best_s = s; best_t = t; }
        }
        return best_t;
    }

public:
    FactWorld() {
        b = std::make_shared<brain2::Brain>(32, 32, 64, 128, 1);
    }

    const std::set<std::string>& actions() const { return _actions; }

    void teach(const std::string& subj, const std::string& rel, const std::string& obj) {
        b->bind_triple(vec(subj), vec(rel), vec(obj));
        if (rel == "requires" || rel == "produces") _actions.insert(subj);
    }

    std::vector<std::pair<std::string, std::string>> facts_of(const std::string& action) {
        auto flat = b->binding.query_all(vec(action), 0.5f);
        std::vector<std::pair<std::string, std::string>> out;
        for (size_t i = 0; i + 1 < flat.size(); i += 2) {
            auto rel = _decode(flat[i]);
            auto obj = _decode(flat[i+1]);
            if (rel && obj) out.push_back({*rel, *obj});
        }
        return out;
    }

    std::set<std::string> requires_for(const std::string& action) {
        std::set<std::string> res;
        for (const auto& f : facts_of(action)) if (f.first == "requires") res.insert(f.second);
        return res;
    }

    std::set<std::string> produces_for(const std::string& action) {
        std::set<std::string> res;
        for (const auto& f : facts_of(action)) if (f.first == "produces") res.insert(f.second);
        return res;
    }
};

class CraftPlanBrain : public SearchProblem<std::set<std::string>> {
private:
    FactWorld* w;
    std::set<std::string> have;
    std::string goal;
public:
    CraftPlanBrain(FactWorld* w, std::set<std::string> have, std::string goal) : w(w), have(have), goal(goal) {}

    std::set<std::string> initial() const override { return have; }

    bool is_goal(const std::set<std::string>& state) const override { return state.count(goal) > 0; }

    std::string key(const std::set<std::string>& state) const {
        std::string k;
        for (const auto& s : state) k += s + ",";
        return k;
    }

    double heuristic(const std::set<std::string>& state) const override { return state.count(goal) > 0 ? 0.0 : 1.0; }

    std::vector<std::tuple<std::string, std::set<std::string>, double>> moves(const std::set<std::string>& state) const override {
        std::vector<std::tuple<std::string, std::set<std::string>, double>> res;
        for (const auto& a : w->actions()) {
            auto req = w->requires_for(a);
            auto prod = w->produces_for(a);
            bool has_all = true;
            for (const auto& r : req) if (!state.count(r)) has_all = false;
            bool new_stuff = false;
            for (const auto& p : prod) if (!state.count(p)) new_stuff = true;
            
            if (has_all && new_stuff) {
                auto ns = state;
                for (const auto& p : prod) ns.insert(p);
                std::string label = a + ": use [";
                for (const auto& r : req) label += r + ",";
                label += "] -> get [";
                for (const auto& p : prod) label += p + ",";
                label += "]";
                res.push_back({label, ns, 1.0});
            }
        }
        return res;
    }
};

}}
