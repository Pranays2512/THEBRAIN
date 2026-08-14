#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace brain3 {
namespace engines {
namespace synthesis {

struct VectorHasher3 {
    std::size_t operator()(const std::vector<std::string>& v) const {
        std::size_t seed = v.size();
        for(const auto& i : v) {
            seed ^= std::hash<std::string>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class SynthesizeGuided : public brain2::reasoning::SearchProblem<std::vector<std::string>, VectorHasher3> {
private:
    std::vector<std::pair<std::string, std::string>> examples;
    int max_len;
    std::map<std::string, double> prior;

    std::vector<std::string> get_words(const std::string& s) const {
        std::istringstream iss(s);
        std::vector<std::string> words;
        std::string w;
        while (iss >> w) words.push_back(w);
        return words;
    }

public:
    std::map<std::string, std::function<std::string(std::string)>> DSL = {
        {"lower", [](std::string s){
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
            return s;
        }},
        {"upper", [](std::string s){
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::toupper(c); });
            return s;
        }},
        {"title", [](std::string s){
            bool new_word = true;
            for (char& c : s) {
                if (std::isspace(c)) { new_word = true; }
                else if (new_word) { c = std::toupper(c); new_word = false; }
                else { c = std::tolower(c); }
            }
            return s;
        }},
        {"capitalize", [](std::string s){
            if (s.empty()) return s;
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
            s[0] = std::toupper(s[0]);
            return s;
        }},
        {"swapcase", [](std::string s){
            for (char& c : s) {
                if (std::islower(c)) c = std::toupper(c);
                else if (std::isupper(c)) c = std::tolower(c);
            }
            return s;
        }},
        {"strip", [](std::string s){
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
            return s;
        }},
        {"no_spaces", [](std::string s){
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        }},
        {"dehyphen", [](std::string s){
            std::replace(s.begin(), s.end(), '-', ' ');
            return s;
        }}
    };

    SynthesizeGuided(const std::vector<std::pair<std::string, std::string>>& ex, int m = 5, const std::map<std::string, double>& p = {})
        : examples(ex), max_len(m), prior(p) {
        
        DSL["first_word"] = [this](std::string s){
            auto w = get_words(s);
            if (w.empty()) throw std::runtime_error("empty");
            return w.front();
        };
        DSL["last_word"] = [this](std::string s){
            auto w = get_words(s);
            if (w.empty()) throw std::runtime_error("empty");
            return w.back();
        };
        DSL["initials"] = [this](std::string s){
            auto w = get_words(s);
            if (w.empty()) throw std::runtime_error("empty");
            std::string res = "";
            for (const auto& word : w) { if (!word.empty()) res += word[0]; }
            return res;
        };
        DSL["reverse_words"] = [this](std::string s){
            auto w = get_words(s);
            std::string res = "";
            for (auto it = w.rbegin(); it != w.rend(); ++it) {
                if (!res.empty()) res += " ";
                res += *it;
            }
            return res;
        };
        DSL["first_char"] = [](std::string s){
            if (s.empty()) throw std::runtime_error("empty");
            return std::string(1, s[0]);
        };
        DSL["last_char"] = [](std::string s){
            if (s.empty()) throw std::runtime_error("empty");
            return std::string(1, s.back());
        };
    }

    std::string run(const std::vector<std::string>& program, std::string s) const {
        for (const auto& op : program) {
            auto it = DSL.find(op);
            if (it != DSL.end()) {
                s = it->second(s);
            }
        }
        return s;
    }

    std::vector<std::string> initial() const override {
        return {};
    }

    bool is_goal(const std::vector<std::string>& prog) const override {
        for (const auto& [inp, out] : examples) {
            try {
                if (run(prog, inp) != out) return false;
            } catch (...) {
                return false;
            }
        }
        return true;
    }

    double heuristic(const std::vector<std::string>& prog) const override {
        return 0.0;
    }

    std::vector<std::tuple<std::string, std::vector<std::string>, double>> moves(const std::vector<std::string>& prog) const override {
        std::vector<std::tuple<std::string, std::vector<std::string>, double>> result;
        if (prog.size() >= max_len) return result;

        for (const auto& [name, fn] : DSL) {
            std::vector<std::string> next_prog = prog;
            next_prog.push_back(name);
            double within = 0.0;
            if (!prior.empty()) {
                auto it = prior.find(name);
                double p = (it != prior.end()) ? it->second : 1e-3;
                within = -std::log(std::max(p, 1e-3));
            }
            result.push_back({"then " + name, next_prog, 1000.0 + within});
        }
        return result;
    }
};

}}}
