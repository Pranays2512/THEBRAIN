#pragma once
#include <vector>
#include <string>
#include <map>
#include <set>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include "../../core/binding_memory.hpp"

namespace brain2 {
namespace knowledge {

using Vector = std::vector<float>;

class KnowledgeEngine {
public:
    int n_dims;
    BindingMemory bm;
    std::map<std::string, Vector> vecs;
    std::set<std::string> entities;
    std::vector<std::tuple<std::string, std::string, std::string>> facts;
    std::map<std::pair<std::string, std::string>, std::string> fact_dict;
    
    float MATCH_THRESHOLD = 0.7f;
    
    KnowledgeEngine(int dims = 64) : n_dims(dims), bm(dims, 1000) {}
    
    Vector get_vec(const std::string& token) {
        if (vecs.count(token)) return vecs[token];
        std::mt19937 gen(std::hash<std::string>{}(token));
        std::normal_distribution<float> dist(0.0f, 1.0f);
        Vector v(n_dims);
        float norm = 0.0f;
        for (int i = 0; i < n_dims; ++i) {
            v[i] = dist(gen);
            norm += v[i] * v[i];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (auto& x : v) x /= norm;
        }
        vecs[token] = v;
        return v;
    }
    
    std::string norm(const std::string& token) {
        std::string t = token;
        t.erase(t.find_last_not_of(" \n\r\t") + 1);
        t.erase(0, t.find_first_not_of(" \n\r\t"));
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        return t;
    }
    
    bool learn(const std::string& subj_raw, const std::string& rel_raw, const std::string& obj_raw) {
        std::string s = norm(subj_raw);
        std::string r = norm(rel_raw);
        std::string o = norm(obj_raw);
        
        auto key = std::make_pair(s, r);
        if (fact_dict.count(key) && fact_dict[key] == o) return false;
        
        bm.bind(get_vec(s), get_vec(r), get_vec(o));
        entities.insert(s);
        entities.insert(o);
        facts.push_back({s, r, o});
        fact_dict[key] = o;
        return true;
    }
    
    std::pair<std::string, float> decode(const Vector& vec) {
        if (entities.empty()) return {"", 0.0f};
        std::string best = "";
        float best_sim = -2.0f;
        
        for (const auto& name : entities) {
            Vector nv = get_vec(name);
            float dot = 0.0f;
            for (int i = 0; i < n_dims; ++i) dot += vec[i] * nv[i];
            if (dot > best_sim) {
                best_sim = dot;
                best = name;
            }
        }
        return {best, best_sim};
    }
    
    std::pair<std::string, float> ask(const std::string& subj_raw, const std::string& rel_raw, int hops = 1, float threshold = 0.7f) {
        std::string s = norm(subj_raw);
        std::string r = norm(rel_raw);
        
        auto key = std::make_pair(s, r);
        if (fact_dict.count(key)) return {fact_dict[key], 1.0f};
        if (entities.count(s) == 0) return {"", 0.0f};
        
        auto res = bm.query(get_vec(s), get_vec(r));
        if (res.second < threshold) return {"", 0.0f}; // Fallback for hops logic if we implement multi-hop in core BM
        
        auto decoded = decode(res.first);
        if (decoded.first == s || decoded.first == "") return {"", 0.0f};
        return {decoded.first, res.second}; // use confidence
    }
    
    std::vector<std::string> derive(const std::string& subj_raw, const std::string& rel_raw, int max_hops = 8) {
        std::string s = norm(subj_raw);
        std::string r = norm(rel_raw);
        
        std::vector<std::string> chain = {s};
        std::string cur = s;
        std::set<std::string> seen = {s};
        
        for (int i = 0; i < max_hops; ++i) {
            auto res = bm.query(get_vec(cur), get_vec(r));
            auto nxt = decode(res.first);
            if (nxt.first == "" || seen.count(nxt.first) || res.second < MATCH_THRESHOLD) break;
            chain.push_back(nxt.first);
            seen.insert(nxt.first);
            cur = nxt.first;
        }
        return chain;
    }
    
    std::string explain(const std::string& subj_raw, const std::string& rel_raw) {
        auto chain = derive(subj_raw, rel_raw);
        if (chain.size() < 2) return "";
        std::string out = "  " + chain[0];
        std::string r = norm(rel_raw);
        for (size_t i = 1; i < chain.size(); ++i) {
            out += " " + r + " " + chain[i];
        }
        return out;
    }
    
    bool knows(const std::string& subj_raw, const std::string& rel_raw, const std::string& obj_raw, int max_hops = 8) {
        std::string o = norm(obj_raw);
        auto chain = derive(subj_raw, rel_raw, max_hops);
        return std::find(chain.begin() + 1, chain.end(), o) != chain.end();
    }
};

} // namespace knowledge
} // namespace brain2
