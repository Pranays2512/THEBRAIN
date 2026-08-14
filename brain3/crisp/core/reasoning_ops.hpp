#pragma once
// reasoning_ops.hpp — the last two proven primitives ported to C++, closing phase 2:
//   cosine_map    : cosine similarity of two sparse feature maps (context_embed.cosine)
//   analogy_score : corresponded-relations count under an entity mapping, with a consistent
//                   relation map (analogy_struct._score) — structure-mapping's core.
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace brain2 {

inline double cosine_map(const std::map<std::string, double>& a,
                         const std::map<std::string, double>& b) {
    double dot = 0, na = 0, nb = 0;
    for (const auto& kv : a) {
        na += kv.second * kv.second;
        auto it = b.find(kv.first);
        if (it != b.end()) dot += kv.second * it->second;
    }
    for (const auto& kv : b) nb += kv.second * kv.second;
    if (na <= 0 || nb <= 0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

using Triple = std::tuple<std::string, std::string, std::string>;  // (subj, rel, obj)

// Corresponded relations under emap, with a consistent relation mapping. Returns the score,
// or -1 if the mapping forces an inconsistent relation correspondence.
inline int analogy_score(const std::vector<Triple>& source,
                         const std::vector<Triple>& target,
                         const std::map<std::string, std::string>& emap) {
    std::map<std::string, std::string> relmap;
    int score = 0;
    for (const auto& s : source) {
        const std::string& a = std::get<0>(s);
        const std::string& r = std::get<1>(s);
        const std::string& b = std::get<2>(s);
        auto ea = emap.find(a), eb = emap.find(b);
        if (ea == emap.end() || eb == emap.end()) continue;
        std::string tr;
        bool found = false;
        for (const auto& t : target) {
            if (std::get<0>(t) == ea->second && std::get<2>(t) == eb->second) {
                tr = std::get<1>(t); found = true; break;
            }
        }
        if (!found) continue;
        auto it = relmap.find(r);
        if (it != relmap.end() && it->second != tr) return -1;   // inconsistent
        relmap[r] = tr;
        score++;
    }
    return score;
}

}  // namespace brain2
