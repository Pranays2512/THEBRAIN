#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>

namespace brain3 {
namespace engines {
namespace synthesis {

class DecisionTree {
public:
    int n_ops;
    int max_depth;
    int min_samples;

    DecisionTree(int ops = 10, int max_d = 10, int min_s = 15)
        : n_ops(ops), max_depth(max_d), min_samples(min_s) {}

    // Dummy prediction for synthesis
    std::vector<double> predict_dist(const std::vector<double>& features) {
        std::vector<double> dist(n_ops, 1.0 / n_ops);
        return dist;
    }
};

struct VectorHasher {
    std::size_t operator()(const std::vector<std::string>& v) const {
        std::size_t seed = v.size();
        for(auto& i : v) {
            seed ^= std::hash<std::string>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class TreeSynth : public brain2::reasoning::SearchProblem<std::vector<std::string>, VectorHasher> {
private:
    std::vector<std::pair<std::string, std::string>> examples;
    DecisionTree* tree;
    int max_len;
    std::vector<std::string> OPS = {"upper", "lower", "title", "strip", "split", "join", "replace", "slice"};

public:
    TreeSynth(const std::vector<std::pair<std::string, std::string>>& ex, DecisionTree* t, int m = 6)
        : examples(ex), tree(t), max_len(m) {}

    std::vector<std::string> initial() const override {
        return {};
    }

    bool is_goal(const std::vector<std::string>& prog) const override {
        return prog.size() > 2; 
    }

    double heuristic(const std::vector<std::string>& prog) const override {
        return 0.0;
    }

    std::vector<std::tuple<std::string, std::vector<std::string>, double>> moves(const std::vector<std::string>& prog) const override {
        std::vector<std::tuple<std::string, std::vector<std::string>, double>> result;
        if (prog.size() >= max_len) return result;

        // Dummy feature extraction and prediction
        std::vector<double> feats(10, 0.5);
        std::vector<double> dist = tree->predict_dist(feats);

        for (size_t i = 0; i < OPS.size(); i++) {
            std::vector<std::string> next_prog = prog;
            next_prog.push_back(OPS[i]);
            
            double cost = 1.0 - std::log(std::max(dist[i], 1e-3));
            result.push_back({"then " + OPS[i], next_prog, cost});
        }
        return result;
    }
};

}}}
