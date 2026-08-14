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
#include <fstream>
#include <sstream>
#include "../../core/binding_memory.hpp"

namespace brain2 {
namespace knowledge {

using Vector = std::vector<float>;

class SemanticMemory {
public:
    int n_dims;
    BindingMemory bm;
    std::map<std::string, Vector> glove;
    std::set<std::string> tokens;
    std::vector<std::tuple<std::string, std::string, std::string>> facts;
    
    float MATCH_THRESHOLD = 0.8f;
    
    SemanticMemory(int dims = 50) : n_dims(dims), bm(dims, 1000) {}
    
    // Load a lightweight dummy GloVe or an actual subset for testing
    void load_glove(const std::map<std::string, Vector>& g) {
        glove = g;
    }
    
    Vector get_vec(const std::string& token) {
        std::string t = token;
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        if (glove.count(t)) return glove[t];
        
        // OOV -> random
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
        return v;
    }
    
    std::string norm(const std::string& token) {
        std::string t = token;
        t.erase(t.find_last_not_of(" \n\r\t") + 1);
        t.erase(0, t.find_first_not_of(" \n\r\t"));
        return t;
    }
    
    bool learn(const std::string& subj_raw, const std::string& rel_raw, const std::string& obj_raw) {
        std::string s = norm(subj_raw);
        std::string r = norm(rel_raw);
        std::string o = norm(obj_raw);
        
        auto tup = std::make_tuple(s, r, o);
        if (std::find(facts.begin(), facts.end(), tup) != facts.end()) return false;
        
        bm.bind(get_vec(s), get_vec(r), get_vec(o));
        tokens.insert(s);
        tokens.insert(o);
        facts.push_back(tup);
        return true;
    }
    
    std::string decode(const Vector& vec) {
        if (tokens.empty()) return "";
        std::string best = "";
        float best_sim = -2.0f;
        
        for (const auto& name : tokens) {
            Vector nv = get_vec(name);
            float dot = 0.0f;
            for (int i = 0; i < n_dims; ++i) dot += vec[i] * nv[i];
            if (dot > best_sim) {
                best_sim = dot;
                best = name;
            }
        }
        return best;
    }
    
    std::pair<std::string, float> ask(const std::string& subj_raw, const std::string& rel_raw) {
        std::string s = norm(subj_raw);
        std::string r = norm(rel_raw);
        
        auto res = bm.query(get_vec(s), get_vec(r));
        if (res.second < MATCH_THRESHOLD) return {"", 0.0f};
        
        std::string tok = decode(res.first);
        return {tok, res.second};
    }
    
    std::vector<std::string> similar(const std::string& token_raw, int k = 5) {
        std::string token = norm(token_raw);
        Vector q = get_vec(token);
        float qn = 0.0f;
        for (float x : q) qn += x * x;
        qn = std::sqrt(qn);
        
        if (qn < 1e-8f || tokens.empty()) return {};
        
        std::vector<std::pair<std::string, float>> sims;
        for (const auto& t : tokens) {
            if (t == token) continue;
            Vector v = get_vec(t);
            float dot = 0.0f;
            float vn = 0.0f;
            for (int i = 0; i < n_dims; ++i) {
                dot += q[i] * v[i];
                vn += v[i] * v[i];
            }
            vn = std::sqrt(vn);
            if (vn > 0.0f) {
                sims.push_back({t, dot / (vn * qn)});
            }
        }
        
        std::sort(sims.begin(), sims.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<std::string> out;
        for (int i = 0; i < std::min((int)sims.size(), k); ++i) {
            out.push_back(sims[i].first);
        }
        return out;
    }
};

} // namespace knowledge
} // namespace brain2
