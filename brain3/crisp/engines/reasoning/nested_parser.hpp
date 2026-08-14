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

class NestedParser {
private:
    std::vector<KnowledgeSource*> sources;
    std::map<std::string, std::pair<std::string, bool>> SUPER = {
        {"heaviest", {"mass", true}}, {"lightest", {"mass", false}},
        {"fastest", {"speed", true}}, {"slowest", {"speed", false}},
        {"densest", {"density", true}}, {"biggest", {"volume", true}}
    };
    std::map<std::string, std::string> SYN = {
        {"velocity", "speed"}, {"weight", "mass"}, {"push", "force"}
    };
    std::set<std::string> ENTS = {"rocket", "sample"};
    std::set<std::string> RELS = {"force", "density", "mass", "speed", "accel", "volume"};

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

    std::pair<std::string, std::string> _resolve_entity(const std::vector<std::string>& toks) const {
        for (const auto& w : toks) {
            if (SUPER.count(w)) {
                std::string rel = SUPER.at(w).first;
                bool is_max = SUPER.at(w).second;
                std::map<std::string, double> vals;
                for (const auto& e : ENTS) {
                    auto v = _solve(Need{e, rel});
                    if (v) vals[e] = *v;
                }
                if (!vals.empty()) {
                    std::string best_e = vals.begin()->first;
                    double best_v = vals.begin()->second;
                    for (const auto& [e, v] : vals) {
                        if (is_max ? (v > best_v) : (v < best_v)) {
                            best_e = e; best_v = v;
                        }
                    }
                    return {best_e, w + " = argmax " + rel};
                }
            }
        }
        for (const auto& w : toks) if (ENTS.count(w)) return {w, ""};
        return {"", ""};
    }

public:
    NestedParser(const std::vector<KnowledgeSource*>& srcs) : sources(srcs) {}

    std::string answer(const std::string& text) const {
        auto toks = _toks(text);
        std::vector<std::string> rels;
        std::set<std::string> seen_rels;
        for (const auto& t : toks) {
            if (RELS.count(t) && !seen_rels.count(t)) {
                rels.push_back(t);
                seen_rels.insert(t);
            }
        }

        auto it_if = std::find(toks.begin(), toks.end(), "if");
        if (it_if != toks.end()) {
            auto it_then = std::find(toks.begin(), toks.end(), "then");
            if (it_then == toks.end()) it_then = toks.end();
            std::vector<std::string> cond(it_if + 1, it_then);
            std::vector<std::string> cons(it_then == toks.end() ? toks.end() : it_then + 1, toks.end());
            
            auto [c_ent, _c_how] = _resolve_entity(cond);
            std::string c_rel;
            for (const auto& t : cond) if (RELS.count(t)) { c_rel = t; break; }
            
            // Extract number
            std::optional<double> num;
            std::string cur;
            for (char c : text) {
                if (isdigit(c) || c == '.') cur += c;
                else if (!cur.empty()) { num = std::stod(cur); break; }
            }
            if (!num && !cur.empty()) num = std::stod(cur);

            bool gt = false;
            std::set<std::string> GT_WORDS = {"greater", "more", "above", "bigger"};
            for (const auto& w : cond) if (GT_WORDS.count(w)) gt = true;

            if (c_ent.empty() || c_rel.empty() || !num) return "(abstain - condition not parseable)";
            
            auto lhs = _solve(Need{c_ent, c_rel});
            if (!lhs) return "(abstain - condition doesn't verify)";
            
            bool holds = gt ? (*lhs > *num) : (*lhs < *num);
            if (!holds) {
                char buf[256];
                snprintf(buf, sizeof(buf), "condition FALSE (%s.%s=%.4g) -> no answer", c_ent.c_str(), c_rel.c_str(), *lhs);
                return std::string(buf);
            }
            
            auto [cons_ent, _cons_how] = _resolve_entity(cons);
            if (cons_ent.empty()) cons_ent = c_ent;
            std::string cons_rel;
            for (const auto& t : cons) if (RELS.count(t)) { cons_rel = t; break; }
            
            if (cons_rel.empty()) return "condition true; consequent doesn't verify -> abstain";
            
            auto v = _solve(Need{cons_ent, cons_rel});
            if (!v) return "condition true; consequent doesn't verify -> abstain";
            
            char buf[256];
            snprintf(buf, sizeof(buf), "condition TRUE (%s.%s=%.4g) -> %s.%s = %.4g", c_ent.c_str(), c_rel.c_str(), *lhs, cons_ent.c_str(), cons_rel.c_str(), *v);
            return std::string(buf);
        }

        auto [ent, how] = _resolve_entity(toks);
        if (!ent.empty() && !rels.empty()) {
            auto v = _solve(Need{ent, rels[0]});
            if (!v) return "(abstain - doesn't verify)";
            std::string sub = how.empty() ? "" : "  [" + how + " -> " + ent + "]";
            char buf[256];
            snprintf(buf, sizeof(buf), "%s.%s = %.4g%s", ent.c_str(), rels[0].c_str(), *v, sub.c_str());
            return std::string(buf);
        }
        return "(abstain - unparseable)";
    }
};

}}
