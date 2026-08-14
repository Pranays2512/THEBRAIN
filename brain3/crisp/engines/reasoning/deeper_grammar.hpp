#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <algorithm>

#include "crisp/engines/reasoning/means_ends.hpp"

namespace brain2 {
namespace reasoning {

class DeeperParser {
private:
    std::vector<KnowledgeSource*> sources;
    std::map<std::string, std::string> SYN = {
        {"velocity", "speed"}, {"weight", "mass"}, {"push", "force"}
    };
    std::set<std::string> ENTS = {"rocket", "sample"};
    std::set<std::string> RELS = {"force", "density", "mass", "speed", "accel", "volume"};
    std::set<std::string> GT = {"greater", "more", "above", "bigger"};

    std::vector<std::string> _toks(const std::string& text) const {
        std::vector<std::string> words;
        std::string cur;
        for (char c : text) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
                cur += std::tolower(c);
            } else if (!cur.empty()) {
                words.push_back(SYN.count(cur) ? SYN.at(cur) : cur);
                cur.clear();
            }
        }
        if (!cur.empty()) words.push_back(SYN.count(cur) ? SYN.at(cur) : cur);
        return words;
    }

    std::optional<double> _solve(const Need& need) const {
        MeansEndsSolver solver(sources);
        return solver.solve(need);
    }

    std::pair<std::optional<bool>, std::string> _eval_atom(const std::vector<std::string>& atom_toks, std::vector<double>& nums, const std::string& default_ent) const {
        std::string ent = default_ent;
        for (const auto& t : atom_toks) if (ENTS.count(t)) { ent = t; break; }
        
        std::string rel = "";
        for (const auto& t : atom_toks) if (RELS.count(t)) { rel = t; break; }
        
        if (ent.empty() || rel.empty() || nums.empty()) return {std::nullopt, ent};
        double num = nums.front(); nums.erase(nums.begin());
        
        auto lhs = _solve(Need{ent, rel});
        if (!lhs) return {std::nullopt, ent};
        
        bool gt = false;
        for (const auto& t : atom_toks) if (GT.count(t)) { gt = true; break; }
        return {gt ? (*lhs > num) : (*lhs < num), ent};
    }

    std::pair<std::optional<bool>, std::string> _eval_condition(const std::vector<std::string>& cond_toks, const std::string& text) const {
        std::vector<double> nums;
        std::string cur;
        for (char c : text) {
            if (isdigit(c) || c == '.') cur += c;
            else if (!cur.empty()) { nums.push_back(std::stod(cur)); cur.clear(); }
        }
        if (!cur.empty()) nums.push_back(std::stod(cur));

        std::string op = "and";
        for (const auto& t : cond_toks) {
            if (t == "and") op = "and";
            else if (t == "or") op = "or";
        }

        std::vector<std::vector<std::string>> parts;
        std::vector<std::string> current_part;
        for (const auto& t : cond_toks) {
            if (t == "and" || t == "or") {
                parts.push_back(current_part);
                current_part.clear();
            } else {
                current_part.push_back(t);
            }
        }
        parts.push_back(current_part);

        std::vector<bool> results;
        std::string ent = "";
        for (const auto& p : parts) {
            auto [r, e] = _eval_atom(p, nums, ent);
            if (!r) return {std::nullopt, e};
            results.push_back(*r);
            ent = e;
        }

        if (op == "or") {
            for (bool b : results) if (b) return {true, ent};
            return {false, ent};
        }
        for (bool b : results) if (!b) return {false, ent};
        return {true, ent};
    }

public:
    DeeperParser(const std::vector<KnowledgeSource*>& srcs) : sources(srcs) {}

    std::string answer(const std::string& text) const {
        auto toks = _toks(text);
        auto it_if = std::find(toks.begin(), toks.end(), "if");
        if (it_if == toks.end()) return "(not a conditional)";
        
        auto it_then = std::find(toks.begin(), toks.end(), "then");
        if (it_then == toks.end()) it_then = toks.end();
        std::vector<std::string> cond(it_if + 1, it_then);
        std::vector<std::string> cons(it_then == toks.end() ? toks.end() : it_then + 1, toks.end());
        
        auto [holds, c_ent] = _eval_condition(cond, text.substr(text.find("if") + 2, (text.find("then") != std::string::npos ? text.find("then") : text.size()) - text.find("if") - 2));
        if (!holds) return "(abstain - condition doesn't parse/verify)";
        if (!*holds) return "condition FALSE -> no answer";

        std::string cons_ent = c_ent;
        for (const auto& t : cons) if (ENTS.count(t)) { cons_ent = t; break; }
        
        std::string cons_rel = "";
        for (const auto& t : cons) if (RELS.count(t)) { cons_rel = t; break; }
        
        if (cons_rel.empty()) return "condition TRUE; consequent doesn't verify -> abstain";
        auto v = _solve(Need{cons_ent, cons_rel});
        if (!v) return "condition TRUE; consequent doesn't verify -> abstain";
        
        char buf[256];
        snprintf(buf, sizeof(buf), "condition TRUE -> %s.%s = %.4g", cons_ent.c_str(), cons_rel.c_str(), *v);
        return std::string(buf);
    }
};

}}
