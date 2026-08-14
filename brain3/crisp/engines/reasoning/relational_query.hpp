#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <memory>
#include <regex>
#include <sstream>

#include "crisp/engines/reasoning/means_ends.hpp"

namespace brain2 {
namespace reasoning {

class RelationalParser {
private:
    std::set<std::string> entities;
    std::set<std::string> LESS = {"lighter", "slower", "smaller", "weaker", "lower", "shorter", "cooler", "cheaper", "thinner", "fewer", "less"};
    std::map<std::string, std::string> ADJ_RELATION = {
        {"heavy", "mass"}, {"light", "mass"}, {"massive", "mass"},
        {"fast", "speed"}, {"quick", "speed"}, {"swift", "speed"}, {"slow", "speed"},
        {"dense", "density"}, {"big", "volume"}, {"large", "volume"}, {"small", "volume"},
        {"strong", "force"}, {"weak", "force"}
    };

    std::string base_adjective(const std::string& w) const {
        if (w.size() > 3 && w.substr(w.size()-3) == "ier") return w.substr(0, w.size()-3) + "y";
        if (w.size() > 2 && w.substr(w.size()-2) == "er") return w.substr(0, w.size()-2);
        return w;
    }

    std::vector<std::string> tokenize(const std::string& text) const {
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
    RelationalParser(const std::set<std::string>& ents) : entities(ents) {}

    std::tuple<std::string, std::string, std::string, int> parse(const std::string& sentence) const {
        auto toks = tokenize(sentence);
        std::vector<std::string> ents;
        for (const auto& t : toks) if (entities.count(t)) ents.push_back(t);
        
        if (ents.size() < 2) return {"", "", "", 0};
        std::string x = ents[0], y = ents[1];

        for (const auto& t : toks) {
            if (t.size() > 2 && t.substr(t.size()-2) == "er" && !entities.count(t)) {
                int dir = LESS.count(t) ? -1 : 1;
                std::string base = base_adjective(t);
                std::string rel = ADJ_RELATION.count(base) ? ADJ_RELATION.at(base) : (ADJ_RELATION.count(t) ? ADJ_RELATION.at(t) : base);
                return {x, y, rel, dir};
            }
        }

        bool has_more = std::find(toks.begin(), toks.end(), "more") != toks.end();
        bool has_less = std::find(toks.begin(), toks.end(), "less") != toks.end();
        if (has_more || has_less) {
            int dir = has_less ? -1 : 1;
            for (const auto& t : toks) {
                if (!entities.count(t) && t != "more" && t != "less" && t != "than" && t != "is" && t != "the" && t != "a") {
                    std::string rel = ADJ_RELATION.count(t) ? ADJ_RELATION.at(t) : t;
                    return {x, y, rel, dir};
                }
            }
        }
        
        return {"", "", "", 0};
    }

    std::string answer(const std::string& sentence, ReasoningEngine* kb, PolicyMemory* mem) {
        auto [x, y, rel, d] = parse(sentence);
        if (x.empty()) return "unparseable";

        FactSource fs(kb);
        PolicySource ps(mem);
        MeansEndsSolver solver({&fs, &ps});

        auto vx = solver.solve(Need{x, rel});
        auto vy = solver.solve(Need{y, rel});

        if (!vx || !vy) return "can't compare — missing " + rel;
        
        bool more = d > 0 ? (*vx > *vy) : (*vx < *vy);
        std::string verdict = more ? "Yes" : "No";
        std::string sign = *vx > *vy ? ">" : (*vx < *vy ? "<" : "=");
        
        char buf[256];
        snprintf(buf, sizeof(buf), "%s. %s %s=%.4g, %s %s=%.4g  (%s)", verdict.c_str(), x.c_str(), rel.c_str(), *vx, y.c_str(), rel.c_str(), *vy, sign.c_str());
        return std::string(buf);
    }
};

} // namespace reasoning
} // namespace brain2
