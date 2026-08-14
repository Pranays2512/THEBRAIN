#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include "crisp/engines/reasoning/means_ends.hpp"

namespace brain2 {
namespace reasoning {

// ── StructuralParser ────────────────────────────────────────────────────────
class StructuralParser {
private:
    std::set<std::string> entities, relations;
    std::map<std::string, std::string> ctx;

    std::vector<std::string> _norm(const std::vector<std::string>& toks) const {
        std::map<std::string, std::string> syn = {
            {"velocity", "speed"}, {"weight", "mass"}, {"push", "force"}, {"pace", "speed"}
        };
        std::vector<std::string> out;
        for (const auto& t : toks) {
            std::string t_syn = syn.count(t) ? syn[t] : t;
            out.push_back(ctx.count(t_syn) ? ctx.at(t_syn) : t_syn);
        }
        return out;
    }

    std::vector<std::string> extract_words(const std::string& text) const {
        std::vector<std::string> words;
        std::string cur;
        for (char c : text) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
                cur += std::tolower(c);
            } else if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) words.push_back(cur);
        return words;
    }

public:
    StructuralParser(std::set<std::string> e, std::set<std::string> r, std::map<std::string, std::string> c = {})
        : entities(e), relations(r), ctx(c) {}

    struct Query {
        std::string kind;
        std::vector<std::string> q_entities;
        std::vector<std::string> q_relations;
    };

    Query parse(const std::string& text) const {
        auto toks = _norm(extract_words(text));
        std::set<std::string> ts(toks.begin(), toks.end());
        std::vector<std::string> ents;
        for (const auto& t : toks) if (entities.count(t)) ents.push_back(t);
        
        std::vector<std::string> rels;
        std::set<std::string> seen_rels;
        for (const auto& t : toks) {
            if (relations.count(t) && !seen_rels.count(t)) {
                rels.push_back(t);
                seen_rels.insert(t);
            }
        }

        std::set<std::string> compare = {"greater", "more", "less", "bigger", "smaller", "heavier", "lighter", "most", "than"};
        bool has_comp = false;
        for (const auto& c : compare) if (ts.count(c)) has_comp = true;

        if (has_comp && ents.size() >= 2 && !rels.empty()) {
            return {"compare", {ents[0], ents[1]}, {rels[0]}};
        }
        if (rels.size() >= 2 && !ents.empty()) {
            return {"compound", {ents[0]}, rels};
        }
        if (!ents.empty() && !rels.empty()) {
            return {"single", {ents[0]}, {rels[0]}};
        }
        return {"unknown", {}, {}};
    }
};

}}
