#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace brain3 {
namespace engines {
namespace synthesis {

struct VH_SE {
    std::size_t operator()(const std::vector<std::string>& v) const {
        std::size_t seed = v.size();
        for (const auto& i : v)
            seed ^= std::hash<std::string>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct SynthesisResult {
    bool found;
    std::vector<std::string> program;

    std::string source() const {
        if (!found) return "(no program found)";
        if (program.empty()) return "(identity)";
        std::string s = "";
        for (size_t i = 0; i < program.size(); i++) {
            if (i > 0) s += " -> ";
            s += program[i];
        }
        return s;
    }
};

class SynthesizeInternal : public brain2::reasoning::SearchProblem<std::vector<std::string>, VH_SE> {
private:
    std::vector<std::pair<std::string, std::string>> examples;
    int max_len;

    std::vector<std::string> split_words(const std::string& s) const {
        std::istringstream iss(s);
        std::vector<std::string> words;
        std::string w;
        while (iss >> w) words.push_back(w);
        return words;
    }

public:
    std::map<std::string, std::function<std::string(std::string)>> DSL;

    SynthesizeInternal(const std::vector<std::pair<std::string, std::string>>& ex, int ml = 4)
        : examples(ex), max_len(ml) {
        DSL["lower"] = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
            return s;
        };
        DSL["upper"] = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::toupper(c); });
            return s;
        };
        DSL["title"] = [](std::string s) {
            bool nw = true;
            for (char& c : s) {
                if (std::isspace(c)) nw = true;
                else if (nw) { c = std::toupper(c); nw = false; }
                else c = std::tolower(c);
            }
            return s;
        };
        DSL["capitalize"] = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
            if (!s.empty()) s[0] = std::toupper(s[0]);
            return s;
        };
        DSL["strip"] = [](std::string s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){ return !std::isspace(ch); }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
            return s;
        };
        DSL["no_spaces"] = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        };
        DSL["first_word"] = [this](std::string s) {
            auto w = split_words(s);
            if (w.empty()) throw std::runtime_error("empty");
            return w.front();
        };
        DSL["last_word"] = [this](std::string s) {
            auto w = split_words(s);
            if (w.empty()) throw std::runtime_error("empty");
            return w.back();
        };
        DSL["initials"] = [this](std::string s) {
            auto w = split_words(s);
            if (w.empty()) throw std::runtime_error("empty");
            std::string res;
            for (const auto& ww : w) if (!ww.empty()) res += ww[0];
            return res;
        };
        DSL["reverse_words"] = [this](std::string s) {
            auto w = split_words(s);
            std::string res;
            for (auto it = w.rbegin(); it != w.rend(); ++it) {
                if (!res.empty()) res += " ";
                res += *it;
            }
            return res;
        };
    }

    std::string run(const std::vector<std::string>& prog, std::string s) const {
        for (const auto& op : prog) {
            auto it = DSL.find(op);
            if (it == DSL.end()) throw std::runtime_error("unknown op");
            s = it->second(s);
        }
        return s;
    }

    std::vector<std::string> initial() const override { return {}; }

    bool is_goal(const std::vector<std::string>& prog) const override {
        for (const auto& [inp, out] : examples) {
            try { if (run(prog, inp) != out) return false; }
            catch (...) { return false; }
        }
        return true;
    }

    double heuristic(const std::vector<std::string>& prog) const override { return 0.0; }

    std::vector<std::tuple<std::string, std::vector<std::string>, double>>
    moves(const std::vector<std::string>& prog) const override {
        std::vector<std::tuple<std::string, std::vector<std::string>, double>> result;
        if ((int)prog.size() >= max_len) return result;
        for (const auto& [name, fn] : DSL) {
            auto next = prog;
            next.push_back(name);
            result.push_back({name, next, 1.0});
        }
        return result;
    }
};

class SynthesisEngine {
private:
    int max_len;

public:
    explicit SynthesisEngine(int ml = 4) {
        if (ml < 1) throw std::invalid_argument("max_len must be a positive integer");
        max_len = ml;
    }

    SynthesisResult synthesize(const std::vector<std::pair<std::string, std::string>>& examples,
                               int max_nodes = 200000) {
        if (examples.empty()) throw std::invalid_argument("need at least one example");
        SynthesizeInternal prob(examples, max_len);
        auto result = brain2::reasoning::solve_astar(prob, max_nodes);
        if (!result.solved) return {false, {}};
        if (result.path.empty()) return {true, {}};  // identity
        return {true, result.path.back().second};
    }
};

}}}
