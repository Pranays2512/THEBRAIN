#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <tuple>
#include <iostream>

namespace brain2 {
namespace reasoning {

// Fact storage representing Knowledge layer
struct Fact {
    std::string subj, rel, obj;
    bool operator==(const Fact& o) const { return subj == o.subj && rel == o.rel && obj == o.obj; }
    bool operator<(const Fact& o) const { 
        if(subj != o.subj) return subj < o.subj;
        if(rel != o.rel) return rel < o.rel;
        return obj < o.obj;
    }
};

class ReasoningEngine {
public:
    std::set<Fact> facts;
    std::vector<std::tuple<std::string, std::string, std::string>> rules;
    std::vector<std::tuple<std::string, std::string, std::string, std::string>> implication_rules;
    std::set<std::string> transitive_rels;
    
    void learn(const std::string& s, const std::string& r, const std::string& o) {
        facts.insert({s, r, o});
    }
    
    void add_rule(const std::string& p1, const std::string& p2, const std::string& concl) {
        rules.push_back({p1, p2, concl});
    }

    void add_implication(const std::string& p_rel, const std::string& p_obj, const std::string& c_rel, const std::string& c_obj) {
        implication_rules.push_back({p_rel, p_obj, c_rel, c_obj});
    }
    
    void set_transitive(const std::string& rel) {
        transitive_rels.insert(rel);
    }
    
    // Transitive closure BFS graph traversal over facts
    std::map<std::string, std::vector<std::string>> closure(const std::string& subj, const std::string& rel, int max_nodes = 100000) {
        std::map<std::string, std::vector<std::string>> adj;
        for (const auto& f : facts) {
            if (f.rel == rel) adj[f.subj].push_back(f.obj);
        }
        
        std::map<std::string, std::vector<std::string>> paths;
        paths[subj] = {subj};
        
        std::queue<std::string> frontier;
        frontier.push(subj);
        
        while (!frontier.empty() && paths.size() <= max_nodes) {
            std::string node = frontier.front();
            frontier.pop();
            
            for (const auto& nbr : adj[node]) {
                if (!paths.count(nbr)) {
                    paths[nbr] = paths[node];
                    paths[nbr].push_back(nbr);
                    frontier.push(nbr);
                }
            }
        }
        
        paths.erase(subj); // report ancestors, not start node
        return paths;
    }
    
    // Backward chaining inference
    std::pair<std::string, std::string> ask(const std::string& subj, const std::string& rel, int max_depth = 8) {
        std::set<std::pair<std::string, std::string>> seen;
        return _ask(subj, rel, max_depth, seen);
    }

private:
    std::pair<std::string, std::string> _ask(const std::string& subj, const std::string& rel, int depth, std::set<std::pair<std::string, std::string>>& seen) {
        if (depth <= 0 || seen.count({subj, rel})) return {"", ""};
        seen.insert({subj, rel});
        
        // 1. Direct facts.
        //
        // These are checked FIRST. Previously the transitive branch below ran
        // ahead of them, so an explicitly taught fact lost to an inferred one:
        // with `zorblat isa quixel` in the store, ask("zorblat","isa") skipped
        // it and answered from the closure instead. The most specific fact
        // available must win, or teaching something has no effect on what the
        // engine says about it.
        for (const auto& f : facts) {
            if (f.subj == subj && f.rel == rel) {
                if (f.obj == "<EXCEPTION>") return {"", "direct exception"};
                return {f.obj, subj + " " + rel + " " + f.obj + " (direct)"};
            }
        }

        // 2. Transitive closure — the NEAREST ancestor, chosen deterministically.
        //
        // This used to return paths.begin()->first. closure() returns a
        // std::map, so begin() is the LEXICOGRAPHICALLY SMALLEST ancestor, not
        // the nearest or the most general. For the chain
        //   zorblat -> quixel -> florn -> grunth -> wibbet -> thingamajig
        // the ancestors sort to florn, grunth, quixel, thingamajig, wibbet, so
        // "what is zorblat?" answered `florn` — a mid-chain node that is neither
        // the parent nor the root. Inserting an ancestor named "aardvark" would
        // have silently changed the answer. Callers cannot build on that.
        //
        // closure() already records the full path to each ancestor, so hop
        // distance was available and simply unused. Nearest wins; ties break
        // lexicographically so the result is stable rather than merely defined.
        if (transitive_rels.count(rel)) {
            auto paths = closure(subj, rel);
            if (!paths.empty()) {
                auto best = paths.begin();
                for (auto it = paths.begin(); it != paths.end(); ++it) {
                    if (it->second.size() < best->second.size() ||
                        (it->second.size() == best->second.size() && it->first < best->first))
                        best = it;
                }
                const size_t hops = best->second.empty() ? 0 : best->second.size() - 1;
                return {best->first,
                        subj + " " + rel + " " + best->first +
                        " (transitive, " + std::to_string(hops) + " hops)"};
            }
        }
        
        // 3. Composition rules
        for (const auto& rule : rules) {
            if (std::get<2>(rule) != rel) continue;
            
            auto mid_ans = _ask(subj, std::get<0>(rule), depth - 1, seen);
            if (mid_ans.first.empty()) continue;
            
            auto z_ans = _ask(mid_ans.first, std::get<1>(rule), depth - 1, seen);
            if (!z_ans.first.empty()) {
                return {z_ans.first, "rule " + std::get<0>(rule) + " + " + std::get<1>(rule)};
            }
        }
        
        // 4. Implication rules (Lossless Compression / Default Logic)
        // Since this is AFTER direct facts (Step 2), explicit exceptions take precedence!
        for (const auto& rule : implication_rules) {
            if (std::get<2>(rule) != rel) continue;
            
            auto premise_ans = _ask(subj, std::get<0>(rule), depth - 1, seen);
            if (premise_ans.first == std::get<1>(rule)) {
                return {std::get<3>(rule), "implied by " + std::get<0>(rule) + "=" + std::get<1>(rule)};
            }
        }
        
        return {"", ""};
    }
};

} // namespace reasoning
} // namespace brain2
