#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <regex>
#include <cmath>
#include <algorithm>

namespace brain2 {
namespace grounding {

using CoocMatrix = std::map<std::string, std::map<std::string, double>>;

inline CoocMatrix build_context_embeddings(const std::vector<std::string>& corpus, int window = 2) {
    CoocMatrix vecs;
    std::regex word_regex("[a-z]+");
    
    for (const auto& line : corpus) {
        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
        
        auto words_begin = std::sregex_iterator(lower_line.begin(), lower_line.end(), word_regex);
        auto words_end = std::sregex_iterator();
        
        std::vector<std::string> toks;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            toks.push_back(i->str());
        }
        
        for (int i = 0; i < (int)toks.size(); ++i) {
            std::string w = toks[i];
            int start = std::max(0, i - window);
            int end = std::min((int)toks.size(), i + window + 1);
            for (int j = start; j < end; ++j) {
                if (j != i) vecs[w][toks[j]] += 1.0;
            }
        }
    }
    return vecs;
}

inline double cosine_sim(const std::map<std::string, double>& a, const std::map<std::string, double>& b) {
    std::set<std::string> keys;
    for (const auto& kv : a) keys.insert(kv.first);
    for (const auto& kv : b) keys.insert(kv.first);
    
    double dot = 0.0, norm_a_sq = 0.0, norm_b_sq = 0.0;
    
    for (const auto& k : keys) {
        double va = a.count(k) ? a.at(k) : 0.0;
        double vb = b.count(k) ? b.at(k) : 0.0;
        dot += va * vb;
    }
    
    for (const auto& kv : a) norm_a_sq += kv.second * kv.second;
    for (const auto& kv : b) norm_b_sq += kv.second * kv.second;
    
    double norm_a = std::sqrt(norm_a_sq);
    double norm_b = std::sqrt(norm_b_sq);
    
    return (norm_a > 0.0 && norm_b > 0.0) ? (dot / (norm_a * norm_b)) : 0.0;
}

inline std::pair<std::string, double> nearest_canonical(
    const std::string& word, 
    const std::vector<std::string>& canonicals, 
    const CoocMatrix& vecs) {
    
    if (vecs.count(word) == 0) return {"", 0.0};
    
    std::string best_word = "";
    double best_sim = -1.0;
    
    for (const auto& c : canonicals) {
        if (vecs.count(c) && c != word) {
            double sim = cosine_sim(vecs.at(word), vecs.at(c));
            if (sim > best_sim) {
                best_sim = sim;
                best_word = c;
            }
        }
    }
    
    return {best_word, best_sim > 0.0 ? best_sim : 0.0};
}

} // namespace grounding
} // namespace brain2
