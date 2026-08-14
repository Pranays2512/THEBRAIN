#pragma once
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>
#include <regex>
#include <iostream>
#include <algorithm>

namespace brain2 {
namespace math {

class ProbLM {
public:
    int order;
    // Map: context (vector of words) -> next_word -> count
    std::map<int, std::map<std::vector<std::string>, std::map<std::string, float>>> counts;
    std::vector<std::string> vocab;
    
    ProbLM(int o = 3) : order(o) {}
    
    void train(const std::vector<std::string>& corpus) {
        std::regex word_re(R"([a-z]+)");
        for (const auto& line : corpus) {
            std::string s = line;
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            
            std::vector<std::string> toks = {"<s>"};
            std::sregex_token_iterator it(s.begin(), s.end(), word_re);
            std::sregex_token_iterator end;
            for (; it != end; ++it) toks.push_back(*it);
            toks.push_back("</s>");
            
            for (const auto& t : toks) {
                if (std::find(vocab.begin(), vocab.end(), t) == vocab.end()) vocab.push_back(t);
            }
            
            for (int n = 0; n < order; ++n) {
                for (size_t i = n; i < toks.size(); ++i) {
                    std::vector<std::string> ctx;
                    for (size_t j = i - n; j < i; ++j) ctx.push_back(toks[j]);
                    counts[n][ctx][toks[i]] += 1.0f;
                }
            }
        }
    }
    
    std::map<std::string, float> dist(const std::vector<std::string>& context) {
        for (int n = std::min(order, (int)context.size() + 1) - 1; n >= 0; --n) {
            std::vector<std::string> ctx;
            if (n > 0) {
                for (size_t i = context.size() - n; i < context.size(); ++i) ctx.push_back(context[i]);
            }
            if (counts[n].count(ctx)) {
                auto tbl = counts[n][ctx];
                float tot = 0.0f;
                for (const auto& kv : tbl) tot += kv.second;
                std::map<std::string, float> probs;
                for (const auto& kv : tbl) probs[kv.first] = kv.second / tot;
                return probs;
            }
        }
        return {};
    }
    
    float entropy(const std::vector<std::string>& context) {
        auto d = dist(context);
        float h = 0.0f;
        for (const auto& kv : d) {
            if (kv.second > 0) h -= kv.second * std::log2(kv.second);
        }
        return h;
    }
    
    std::vector<std::string> generate(std::vector<std::string> seed = {"<s>"}, int max_len = 14, int seed_rng = 0) {
        std::mt19937 rng(seed_rng);
        std::vector<std::string> out = seed;
        
        for (int i = 0; i < max_len; ++i) {
            auto d = dist(out);
            if (d.empty()) break;
            
            std::vector<std::string> words;
            std::vector<float> probs;
            for (const auto& kv : d) {
                words.push_back(kv.first);
                probs.push_back(kv.second);
            }
            
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            std::string nxt = words[dist(rng)];
            
            if (nxt == "</s>") break;
            if (nxt != "<s>") out.push_back(nxt);
        }
        
        std::vector<std::string> final_out;
        for (const auto& w : out) {
            if (w != "<s>" && w != "</s>") final_out.push_back(w);
        }
        return final_out;
    }
};

} // namespace math
} // namespace brain2
