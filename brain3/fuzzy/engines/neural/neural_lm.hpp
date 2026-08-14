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
namespace neural {

// Simple dense matrix helper
struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& at(int r, int c) { return data[r * cols + c]; }
    const float& at(int r, int c) const { return data[r * cols + c]; }
    
    void randomize(std::mt19937& rng, float scale) {
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& v : data) v = dist(rng);
    }
};

class NeuralLM {
public:
    int k, d, h, epochs;
    float lr;
    std::map<std::string, int> w2i;
    std::vector<std::string> i2w;
    
    Matrix E;  // Embeddings: V x d
    Matrix W1; // Hidden: h x (k*d)
    std::vector<float> b1; // h
    Matrix W2; // Out: V x h
    std::vector<float> b2; // V
    
    NeuralLM(int context_k=2, int embed_d=24, int hidden_h=48, float learn_rate=0.3f, int num_epochs=400) 
        : k(context_k), d(embed_d), h(hidden_h), lr(learn_rate), epochs(num_epochs) {}
        
    std::vector<std::string> tokenize(const std::string& line) {
        std::vector<std::string> toks;
        for (int i = 0; i < k; ++i) toks.push_back("<s>");
        
        std::string s = line;
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        std::regex word_re(R"([a-z]+)");
        std::sregex_token_iterator it(s.begin(), s.end(), word_re);
        std::sregex_token_iterator end;
        for (; it != end; ++it) toks.push_back(*it);
        
        toks.push_back("</s>");
        return toks;
    }
    
    void train(const std::vector<std::string>& corpus, int seed=0) {
        std::mt19937 rng(seed);
        std::vector<std::string> unique_words = {"<unk>"};
        
        std::vector<std::vector<std::string>> all_toks;
        for (const auto& line : corpus) {
            auto toks = tokenize(line);
            all_toks.push_back(toks);
            for (const auto& w : toks) {
                if (std::find(unique_words.begin(), unique_words.end(), w) == unique_words.end()) {
                    unique_words.push_back(w);
                }
            }
        }
        
        std::sort(unique_words.begin(), unique_words.end());
        for (size_t i = 0; i < unique_words.size(); ++i) {
            w2i[unique_words[i]] = i;
            i2w.push_back(unique_words[i]);
        }
        int V = unique_words.size();
        
        std::vector<std::pair<std::vector<int>, int>> pairs;
        for (const auto& toks : all_toks) {
            for (size_t i = k; i < toks.size(); ++i) {
                std::vector<int> ctx;
                for (int j = k; j > 0; --j) ctx.push_back(w2i[toks[i - j]]);
                pairs.push_back({ctx, w2i[toks[i]]});
            }
        }
        
        E = Matrix(V, d); E.randomize(rng, 0.1f);
        W1 = Matrix(h, k * d); W1.randomize(rng, 0.1f);
        b1.assign(h, 0.0f);
        W2 = Matrix(V, h); W2.randomize(rng, 0.1f);
        b2.assign(V, 0.0f);
        
        // Very basic SGD loop without batching for simplicity, matching the pure python version
        for (int ep = 0; ep < epochs; ++ep) {
            std::shuffle(pairs.begin(), pairs.end(), rng);
            
            for (const auto& pair : pairs) {
                const auto& ctx = pair.first;
                int tgt = pair.second;
                
                // Forward
                std::vector<float> x(k * d);
                for (int j = 0; j < k; ++j) {
                    for (int m = 0; m < d; ++m) x[j * d + m] = E.at(ctx[j], m);
                }
                
                std::vector<float> hh(h, 0.0f);
                for (int r = 0; r < h; ++r) {
                    float sum = b1[r];
                    for (int c = 0; c < k * d; ++c) sum += W1.at(r, c) * x[c];
                    hh[r] = std::tanh(sum);
                }
                
                std::vector<float> logits(V, 0.0f);
                float max_l = -1e9f;
                for (int r = 0; r < V; ++r) {
                    float sum = b2[r];
                    for (int c = 0; c < h; ++c) sum += W2.at(r, c) * hh[c];
                    logits[r] = sum;
                    if (sum > max_l) max_l = sum;
                }
                
                std::vector<float> p(V, 0.0f);
                float sum_p = 0.0f;
                for (int r = 0; r < V; ++r) {
                    p[r] = std::exp(logits[r] - max_l);
                    sum_p += p[r];
                }
                for (int r = 0; r < V; ++r) p[r] /= sum_p;
                
                // Backward
                std::vector<float> dl = p;
                dl[tgt] -= 1.0f;
                
                std::vector<float> dhh(h, 0.0f);
                for (int c = 0; c < h; ++c) {
                    for (int r = 0; r < V; ++r) dhh[c] += W2.at(r, c) * dl[r];
                }
                
                std::vector<float> dhpre(h);
                for (int r = 0; r < h; ++r) dhpre[r] = dhh[r] * (1.0f - hh[r] * hh[r]);
                
                std::vector<float> dx(k * d, 0.0f);
                for (int c = 0; c < k * d; ++c) {
                    for (int r = 0; r < h; ++r) dx[c] += W1.at(r, c) * dhpre[r];
                }
                
                // Update
                for (int r = 0; r < V; ++r) {
                    b2[r] -= lr * dl[r];
                    for (int c = 0; c < h; ++c) W2.at(r, c) -= lr * dl[r] * hh[c];
                }
                for (int r = 0; r < h; ++r) {
                    b1[r] -= lr * dhpre[r];
                    for (int c = 0; c < k * d; ++c) W1.at(r, c) -= lr * dhpre[r] * x[c];
                }
                for (int j = 0; j < k; ++j) {
                    for (int m = 0; m < d; ++m) E.at(ctx[j], m) -= lr * dx[j * d + m];
                }
            }
        }
    }
    
    std::map<std::string, float> dist(const std::vector<std::string>& context) {
        std::vector<std::string> ctx = context;
        while ((int)ctx.size() < k) ctx.insert(ctx.begin(), "<s>");
        
        std::vector<int> ids;
        for (int i = ctx.size() - k; i < (int)ctx.size(); ++i) {
            if (w2i.count(ctx[i])) ids.push_back(w2i[ctx[i]]);
            else ids.push_back(w2i["<unk>"]);
        }
        
        std::vector<float> x(k * d);
        for (int j = 0; j < k; ++j) {
            for (int m = 0; m < d; ++m) x[j * d + m] = E.at(ids[j], m);
        }
        
        std::vector<float> hh(h, 0.0f);
        for (int r = 0; r < h; ++r) {
            float sum = b1[r];
            for (int c = 0; c < k * d; ++c) sum += W1.at(r, c) * x[c];
            hh[r] = std::tanh(sum);
        }
        
        std::vector<float> logits(V(), 0.0f);
        float max_l = -1e9f;
        for (int r = 0; r < V(); ++r) {
            float sum = b2[r];
            for (int c = 0; c < h; ++c) sum += W2.at(r, c) * hh[c];
            logits[r] = sum;
            if (sum > max_l) max_l = sum;
        }
        
        float sum_p = 0.0f;
        std::map<std::string, float> probs;
        for (int r = 0; r < V(); ++r) {
            float p = std::exp(logits[r] - max_l);
            sum_p += p;
            probs[i2w[r]] = p;
        }
        for (int r = 0; r < V(); ++r) probs[i2w[r]] /= sum_p;
        
        return probs;
    }
    
    int V() const { return i2w.size(); }
    
    std::vector<std::string> generate(int max_len=12, int seed=0) {
        std::mt19937 rng(seed);
        std::vector<std::string> out;
        for (int i = 0; i < max_len; ++i) {
            auto d = dist(out);
            d.erase("<s>");
            d.erase("<unk>");
            
            std::vector<std::string> words;
            std::vector<float> probs;
            float sum = 0.0f;
            for (const auto& kv : d) {
                words.push_back(kv.first);
                probs.push_back(kv.second);
                sum += kv.second;
            }
            if (sum == 0.0f) break;
            
            std::discrete_distribution<int> pdist(probs.begin(), probs.end());
            std::string nxt = words[pdist(rng)];
            if (nxt == "</s>") break;
            out.push_back(nxt);
        }
        return out;
    }
};

} // namespace neural
} // namespace brain2
