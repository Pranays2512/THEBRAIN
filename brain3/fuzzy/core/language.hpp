#pragma once
/*
 * language.hpp — Bidirectional Language, Component 8 of Brain v2
 *
 * Words are learned SOM vectors — same concept space as perception.
 * No hardcoded grammar. No template rules.
 * Grammar emerges from sequence statistics learned in the Predictor.
 *
 * Encoding (word → concept vector):
 *   lookup table: word string → float[n_dims]
 *   Learned: each word's vector drifts toward co-occurring concept activations
 *
 * Decoding (concept vector → word):
 *   nearest-neighbor search in word embedding table
 *   Returns top-k candidate words with similarity scores
 *
 * Inner speech:
 *   WorkingMemory context → decode → emit word → re-encode → observe in SOM
 *   This loop runs autonomously and is what generates thoughts.
 *
 * Learning:
 *   When word heard AND SOM activation present:
 *     word_vec += lr * (som_activation - word_vec)
 *   Word vectors drift toward what the brain perceives when hearing them.
 */

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <random>

namespace brain2 {

struct WordEntry {
    std::vector<float> vec;  // embedding in concept space
    int    frequency;        // how many times word was heard
    float  familiarity;      // smoothed frequency — how well brain "knows" word
};

class Language {
public:
    int   n_dims;
    float lr;             // embedding learning rate

private:
    std::unordered_map<std::string, WordEntry> words_;
    std::map<std::vector<float>, std::string>  reverse_map_;
    std::vector<std::string>                   vocab_;   // ordered list for indexing
    std::vector<float>                         flat_embeddings_; // contiguous matrix for ultra-fast sim
    std::unique_ptr<std::mutex>                mtx_;
    bool                                       frozen_ = false;
    mutable std::mt19937                       rng_;

    // L2 squared distance
    static float l2sq(const std::vector<float>& a,
                      const std::vector<float>& b) noexcept {
        float s = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) { float d = a[i]-b[i]; s += d*d; }
        return s;
    }

    // Cosine similarity
    static float cosine(const std::vector<float>& a,
                        const std::vector<float>& b) noexcept {
        float dot = 0.f, na = 0.f, nb = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    // Initialize new word with small random vector restricted to the Left Hemisphere (Language)
    std::vector<float> random_vec(std::mt19937& rng) const {
        std::normal_distribution<float> dist(0.f, 0.1f);
        std::vector<float> v(n_dims, 0.f);
        // Language only lives in the first half of the brain
        int half = n_dims / 2;
        for (int i = 0; i < half; i++) {
            v[i] = dist(rng);
        }
        return v;
    }

public:
    Language() : n_dims(0), lr(0.05f),
                 mtx_(std::make_unique<std::mutex>()), rng_(1337) {}

    Language(int n_dims, float lr = 0.05f)
        : n_dims(n_dims), lr(lr),
          mtx_(std::make_unique<std::mutex>()), rng_(1337) {}

    Language(Language&&)            = default;
    Language& operator=(Language&&) = default;
    Language(const Language&)       = delete;
    Language& operator=(const Language&) = delete;

    // Register a word with optional initial vector
    // If initial_vec empty: random initialization
    void register_word(const std::string& word,
                       const std::vector<float>& initial_vec = {}) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (words_.find(word) == words_.end()) {
            std::vector<float> v = initial_vec.size() == n_dims ? initial_vec : random_vec(rng_);
            words_[word] = {v, 1, 0.01f};
            vocab_.push_back(word);
            flat_embeddings_.insert(flat_embeddings_.end(), v.begin(), v.end());
            reverse_map_[v] = word;
        }
    }

    // Encode: word → concept vector
    // Dynamically auto-assigns a vector if the word is unknown.
    std::vector<float> encode(const std::string& word) {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = words_.find(word);
        if (it == words_.end()) {
            if (frozen_) return std::vector<float>(n_dims, 0.f);
            
            // Deterministic pseudo-random generation based on the word itself
            // Ensures OOV words always get the exact same representation across runs
            size_t h = std::hash<std::string>{}(word);
            std::mt19937 local_rng(h);
            std::vector<float> v = random_vec(local_rng);
            
            words_[word] = {v, 1, 0.01f};
            vocab_.push_back(word);
            flat_embeddings_.insert(flat_embeddings_.end(), v.begin(), v.end());
            reverse_map_[v] = word;
            return v;
        }
        return it->second.vec;
    }

    // Decode: concept vector → top-k words with similarity scores
    std::vector<std::pair<std::string, float>> decode(const std::vector<float>& vec, int k = 5,
                                                      const std::vector<std::string>& penalize = {},
                                                      int max_scan = -1) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        
        // Pre-compute query norm for super fast SIMD cosine sim
        float na = 0.f;
        for (int i = 0; i < n_dims; i++) na += vec[i] * vec[i];
        float inv_na = (na < 1e-8f) ? 0.f : 1.0f / std::sqrt(na);

        if (k == 1) {
            std::string best_w;
            float best_score = -2.f;
            int count = 0;
            int n_words = vocab_.size();
            for (int i = 0; i < n_words; i++) {
                if (max_scan > 0 && count++ >= max_scan) break;
                if (!penalize.empty() && std::find(penalize.begin(), penalize.end(), vocab_[i]) != penalize.end()) continue;
                
                // Fast contiguous dot product
                float dot = 0.f, nb = 0.f;
                int offset = i * n_dims;
                for (int d = 0; d < n_dims; d++) {
                    float b_val = flat_embeddings_[offset + d];
                    dot += vec[d] * b_val;
                    nb += b_val * b_val;
                }
                float s = 0.f;
                if (nb >= 1e-8f && inv_na > 0.f) s = dot * inv_na / std::sqrt(nb);
                
                if (s > best_score) { best_score = s; best_w = vocab_[i]; }
            }
            if (best_w.empty()) return {};
            return {{best_w, best_score}};
        }

        std::vector<std::pair<std::string, float>> scores;
        int count = 0;
        int n_words = vocab_.size();
        for (int i = 0; i < n_words; i++) {
            if (max_scan > 0 && count++ >= max_scan) break;
            if (!penalize.empty() && std::find(penalize.begin(), penalize.end(), vocab_[i]) != penalize.end()) continue;
            
            float dot = 0.f, nb = 0.f;
            int offset = i * n_dims;
            for (int d = 0; d < n_dims; d++) {
                float b_val = flat_embeddings_[offset + d];
                dot += vec[d] * b_val;
                nb += b_val * b_val;
            }
            float s = 0.f;
            if (nb >= 1e-8f && inv_na > 0.f) s = dot * inv_na / std::sqrt(nb);
            
            scores.push_back({vocab_[i], s});
        }
        int sort_len = std::min(k, (int)scores.size());
        if (sort_len > 0) {
            std::partial_sort(scores.begin(), scores.begin() + sort_len, scores.end(),
                              [](const auto& a, const auto& b) { return a.second > b.second; });
            scores.resize(sort_len);
        }
        return scores;
    }

    // Best single word for a concept vector, with optional repetition penalty
    std::string best_word(const std::vector<float>& concept_vec, const std::vector<std::string>& penalize_words = {}, int max_scan = -1) const {
        {
            std::lock_guard<std::mutex> lock(*mtx_);
            auto it = reverse_map_.find(concept_vec);
            if (it != reverse_map_.end()) {
                if (std::find(penalize_words.begin(), penalize_words.end(), it->second) == penalize_words.end()) {
                    return it->second;
                }
            }
        }
        auto top = decode(concept_vec, 1, penalize_words, max_scan);
        return top.empty() ? "" : top[0].first;
    }

    // Hear a word in context of current SOM activation — updates embedding
    // This is how word meaning is learned: word co-occurs with perception
    void hear(const std::string& word,
              const std::vector<float>& som_activation) {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = words_.find(word);
        if (it == words_.end()) {
            // Auto-register unknown word
            WordEntry e;
            e.vec         = som_activation;  // initialize to current context
            e.frequency   = 0;
            e.familiarity = 0.f;
            words_[word]  = std::move(e);
            vocab_.push_back(word);
            flat_embeddings_.insert(flat_embeddings_.end(), som_activation.begin(), som_activation.end());
            it = words_.find(word);
        }
        auto& e = it->second;
        e.frequency++;
        e.familiarity = 0.9f * e.familiarity + 0.1f;  // smoothed

        // Move embedding toward current activation (Hebbian)
        if (!frozen_ && (int)som_activation.size() == n_dims) {
            auto it_v = std::find(vocab_.begin(), vocab_.end(), word);
            if (it_v != vocab_.end()) {
                int idx = std::distance(vocab_.begin(), it_v);
                int offset = idx * n_dims;
                for (int i = 0; i < n_dims; i++) {
                    float delta = lr * (som_activation[i] - e.vec[i]);
                    e.vec[i] += delta;
                    flat_embeddings_[offset + i] += delta;
                }
            }
        }
    }

    // Speak: generate word sequence from sequence of concept vectors
    // Used for inner speech and output generation
    std::vector<std::string> speak(
            const std::vector<std::vector<float>>& concept_seq,
            float min_sim = 0.0f) const {
        std::vector<std::string> words;
        for (const auto& cv : concept_seq) {
            auto top = decode(cv, 1);
            if (!top.empty() && top[0].second >= min_sim)
                words.push_back(top[0].first);
        }
        return words;
    }

    int   vocab_size()          const noexcept { return (int)words_.size(); }
    
    int word_id(const std::string& w) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = std::find(vocab_.begin(), vocab_.end(), w);
        if (it == vocab_.end()) return -1;
        return std::distance(vocab_.begin(), it);
    }
    
    const float* flat_embeddings_ptr() const { return flat_embeddings_.data(); }

    const bool* is_frozen_ptr() const { return &frozen_; }
    
    void freeze_vocabulary(bool freeze = true) {
        std::lock_guard<std::mutex> lock(*mtx_);
        frozen_ = freeze;
    }
    void set_frozen(bool frozen) {
        std::lock_guard<std::mutex> lock(*mtx_);
        frozen_ = frozen;
    }
    bool is_frozen() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return frozen_;
    }

    void load_semantics(const std::string& filepath) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f) throw std::runtime_error("Language::load_semantics: cannot open " + filepath);
        
        int vocab_size = 0, dim = 0;
        f.read((char*)&vocab_size, sizeof(int));
        f.read((char*)&dim, sizeof(int));
        
        if (dim != n_dims) {
            throw std::runtime_error("Language::load_semantics: dim mismatch. Expected " + std::to_string(n_dims) + " got " + std::to_string(dim));
        }

        std::lock_guard<std::mutex> lock(*mtx_);
        for (int i = 0; i < vocab_size; i++) {
            std::string word;
            char c;
            while (f.get(c) && c != '\0') {
                word += c;
            }
            std::vector<float> vec(n_dims);
            f.read((char*)vec.data(), n_dims * sizeof(float));
            
            words_[word] = {vec, 1, 0.01f};
            vocab_.push_back(word);
            flat_embeddings_.insert(flat_embeddings_.end(), vec.begin(), vec.end());
            reverse_map_[vec] = word;
        }
    }

    bool  knows(const std::string& w) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return words_.count(w) > 0;
    }
    float familiarity(const std::string& w) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = words_.find(w);
        return it == words_.end() ? 0.f : it->second.familiarity;
    }
    int frequency(const std::string& w) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = words_.find(w);
        return it == words_.end() ? 0 : it->second.frequency;
    }

    std::vector<std::string> vocab() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return vocab_;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Language::save: cannot open " + path);
        f.write((const char*)&n_dims, sizeof(int));
        f.write((const char*)&lr,     sizeof(float));
        f.write((const char*)&frozen_,sizeof(bool));
        std::vector<std::string> unique_valid_words;
        std::unordered_set<std::string> seen;
        for (const auto& w : vocab_) {
            if (seen.count(w) == 0 && words_.find(w) != words_.end()) {
                seen.insert(w);
                unique_valid_words.push_back(w);
            }
        }
        
        int n = (int)unique_valid_words.size();
        f.write((const char*)&n, sizeof(int));
        
        for (const auto& w : unique_valid_words) {
            
            const auto& e = words_.at(w);
            int wlen = (int)w.size();
            f.write((const char*)&wlen, sizeof(int));
            f.write(w.data(), wlen);
            f.write((const char*)e.vec.data(),
                    (std::streamsize)(n_dims * sizeof(float)));
            f.write((const char*)&e.frequency,   sizeof(int));
            f.write((const char*)&e.familiarity, sizeof(float));
        }
    }

    static Language load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Language::load: cannot open " + path);
        Language l;
        f.read((char*)&l.n_dims, sizeof(int));
        f.read((char*)&l.lr,     sizeof(float));
        f.read((char*)&l.frozen_,sizeof(bool));
        l.mtx_ = std::make_unique<std::mutex>();
        int n; f.read((char*)&n, sizeof(int));
        for (int i = 0; i < n; i++) {
            int wlen; f.read((char*)&wlen, sizeof(int));
            std::string w(wlen, '\0');
            f.read(w.data(), wlen);
            WordEntry e;
            e.vec.resize(l.n_dims);
            f.read((char*)e.vec.data(),
                   (std::streamsize)(l.n_dims * sizeof(float)));
            f.read((char*)&e.frequency,   sizeof(int));
            f.read((char*)&e.familiarity, sizeof(float));
            l.words_[w] = std::move(e);
            l.vocab_.push_back(w);
            l.flat_embeddings_.insert(l.flat_embeddings_.end(), l.words_[w].vec.begin(), l.words_[w].vec.end());
        }
        return l;
    }

    void expand_dims(int new_dims) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (new_dims <= n_dims) return;
        for (auto& [w, e] : words_) {
            e.vec.resize(new_dims, 0.f);
        }
        n_dims = new_dims;
    }
};

} // namespace brain2
