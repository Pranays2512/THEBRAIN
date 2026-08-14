#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <algorithm>
#include <random>

namespace brain2 {
namespace events {

using Fact = std::tuple<std::string, std::string, std::string>;

inline std::vector<std::string> get_entities(const std::vector<Fact>& triples) {
    std::vector<std::string> es;
    for (const auto& t : triples) {
        if (std::find(es.begin(), es.end(), std::get<0>(t)) == es.end()) es.push_back(std::get<0>(t));
        if (std::find(es.begin(), es.end(), std::get<2>(t)) == es.end()) es.push_back(std::get<2>(t));
    }
    return es;
}

inline std::set<std::string> get_relations(const std::vector<Fact>& triples) {
    std::set<std::string> rels;
    for (const auto& t : triples) rels.insert(std::get<1>(t));
    return rels;
}

// -----------------------------------------------------------------------------
// Analogy Engine: Shared relation vocabulary (analogy_engine.py)
// -----------------------------------------------------------------------------

inline std::set<std::pair<std::string, std::string>> get_signature(
    const std::string& obj, 
    const std::vector<Fact>& facts, 
    const std::set<std::string>& rels) {
    std::set<std::pair<std::string, std::string>> sig;
    for (const auto& f : facts) {
        if (rels.count(std::get<1>(f)) == 0) continue;
        if (std::get<0>(f) == obj) sig.insert({std::get<1>(f), "subj"});
        if (std::get<2>(f) == obj) sig.insert({std::get<1>(f), "obj"});
    }
    return sig;
}

struct SharedAnalogyResult {
    std::map<std::string, std::string> mapping;
    std::vector<std::tuple<Fact, Fact>> transfers; // (predicted_fact, from_source_fact)
};

inline SharedAnalogyResult map_domains_shared(const std::vector<Fact>& source, const std::vector<Fact>& target) {
    auto s_rels = get_relations(source);
    auto t_rels = get_relations(target);
    std::set<std::string> common;
    std::set_intersection(s_rels.begin(), s_rels.end(), t_rels.begin(), t_rels.end(), std::inserter(common, common.begin()));

    auto s_objs = get_entities(source);
    auto t_objs = get_entities(target);

    std::map<std::string, std::set<std::pair<std::string, std::string>>> s_sig, t_sig;
    for (const auto& o : s_objs) s_sig[o] = get_signature(o, source, common);
    for (const auto& o : t_objs) t_sig[o] = get_signature(o, target, common);

    std::map<std::string, std::string> mapping;
    std::set<std::string> used;

    for (const auto& so : s_objs) {
        auto& sig = s_sig[so];
        if (sig.empty()) continue;
        std::vector<std::string> cands;
        for (const auto& to : t_objs) {
            if (t_sig[to] == sig && used.count(to) == 0) cands.push_back(to);
        }
        if (cands.size() == 1) { // unambiguous correspondence only
            mapping[so] = cands[0];
            used.insert(cands[0]);
        }
    }

    std::vector<std::tuple<Fact, Fact>> transfers;
    for (const auto& src_f : source) {
        if (mapping.count(std::get<0>(src_f)) && mapping.count(std::get<2>(src_f))) {
            Fact cand = {mapping[std::get<0>(src_f)], std::get<1>(src_f), mapping[std::get<2>(src_f)]};
            if (std::find(target.begin(), target.end(), cand) == target.end()) {
                transfers.push_back({cand, src_f});
            }
        }
    }
    return {mapping, transfers};
}

// -----------------------------------------------------------------------------
// Structural Analogy: No shared relation vocabulary (analogy_struct.py)
// -----------------------------------------------------------------------------

inline std::pair<int, std::map<std::string, std::string>> score_alignment(
    const std::vector<Fact>& source, 
    const std::vector<Fact>& target, 
    const std::map<std::string, std::string>& emap) {
    
    std::map<std::string, std::string> relmap;
    int score = 0;
    
    for (const auto& src_f : source) {
        auto a = std::get<0>(src_f);
        auto r = std::get<1>(src_f);
        auto b = std::get<2>(src_f);
        
        if (emap.count(a) == 0 || emap.count(b) == 0) continue;
        
        std::vector<std::string> cands;
        for (const auto& tgt_f : target) {
            if (std::get<0>(tgt_f) == emap.at(a) && std::get<2>(tgt_f) == emap.at(b)) {
                cands.push_back(std::get<1>(tgt_f));
            }
        }
        if (cands.empty()) continue;
        
        std::string tr = cands[0];
        if (relmap.count(r) && relmap[r] != tr) return {-1, {}};
        
        bool claimed = false;
        for (const auto& kv : relmap) {
            if (kv.second == tr && kv.first != r) { claimed = true; break; }
        }
        if (claimed) continue; // target relation already claimed
        
        relmap[r] = tr;
        score++;
    }
    return {score, relmap};
}

struct StructAnalogyResult {
    std::map<std::string, std::string> emap;
    std::map<std::string, std::string> relmap;
    int score = -1;
};

inline StructAnalogyResult align_greedy(const std::vector<Fact>& source, const std::vector<Fact>& target, int restarts = 8, unsigned int seed = 0) {
    auto es = get_entities(source);
    auto et = get_entities(target);
    if (es.size() > et.size()) return {};
    
    std::mt19937 rng(seed);
    
    StructAnalogyResult best;
    
    for (int r = 0; r < restarts; r++) {
        std::vector<std::string> perm = et;
        std::shuffle(perm.begin(), perm.end(), rng);
        
        std::map<std::string, std::string> emap;
        for (size_t i = 0; i < es.size(); i++) emap[es[i]] = perm[i];
        
        bool improved = true;
        int cur_score = score_alignment(source, target, emap).first;
        
        while (improved) {
            improved = false;
            for (size_t i = 0; i < es.size(); i++) {
                for (const auto& tgt : et) {
                    if (emap[es[i]] == tgt) continue;
                    
                    std::map<std::string, std::string> em2 = emap;
                    std::string owner = "";
                    for (const auto& kv : em2) {
                        if (kv.second == tgt) { owner = kv.first; break; }
                    }
                    if (!owner.empty()) em2[owner] = emap[es[i]];
                    em2[es[i]] = tgt;
                    
                    int s2 = score_alignment(source, target, em2).first;
                    if (s2 > cur_score) {
                        emap = em2;
                        cur_score = s2;
                        improved = true;
                    }
                }
            }
        }
        
        auto res = score_alignment(source, target, emap);
        if (res.first > best.score) {
            best.score = res.first;
            best.emap = emap;
            best.relmap = res.second;
        }
    }
    return best;
}

inline std::vector<std::pair<Fact, bool>> transfer_structural(
    const std::vector<Fact>& source, 
    const std::vector<Fact>& target, 
    const std::map<std::string, std::string>& emap, 
    const std::map<std::string, std::string>& relmap) {
    
    std::vector<std::pair<Fact, bool>> preds;
    for (const auto& src_f : source) {
        auto a = std::get<0>(src_f);
        auto r = std::get<1>(src_f);
        auto b = std::get<2>(src_f);
        
        if (emap.count(a) && emap.count(b)) {
            std::string tr = relmap.count(r) ? relmap.at(r) : r;
            Fact cand = {emap.at(a), tr, emap.at(b)};
            if (std::find(target.begin(), target.end(), cand) == target.end()) {
                preds.push_back({cand, relmap.count(r) == 0});
            }
        }
    }
    return preds;
}

} // namespace events
} // namespace brain2
