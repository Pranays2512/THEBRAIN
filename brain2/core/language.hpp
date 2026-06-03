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
#include <algorithm>
#include <numeric>
#include <cmath>
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
    std::vector<std::string>                   vocab_;   // ordered list for indexing
    std::unique_ptr<std::mutex>                mtx_;

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

    // Initialize new word with small random vector
    std::vector<float> random_vec(std::mt19937& rng) const {
        std::normal_distribution<float> dist(0.f, 0.1f);
        std::vector<float> v(n_dims);
        for (auto& x : v) x = dist(rng);
        return v;
    }

public:
    Language() : n_dims(0), lr(0.05f),
                 mtx_(std::make_unique<std::mutex>()) {}

    Language(int n_dims, float lr = 0.05f)
        : n_dims(n_dims), lr(lr),
          mtx_(std::make_unique<std::mutex>()) {}

    Language(Language&&)            = default;
    Language& operator=(Language&&) = default;
    Language(const Language&)       = delete;
    Language& operator=(const Language&) = delete;

    // Register a word with optional initial vector
    // If initial_vec empty: random initialization
    void register_word(const std::string& word,
                       const std::vector<float>& initial_vec = {}) {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (words_.count(word)) return;
        WordEntry e;
        if ((int)initial_vec.size() == n_dims) {
            e.vec = initial_vec;
        } else {
            std::mt19937 rng(std::hash<std::string>{}(word));
            e.vec = random_vec(rng);
        }
        e.frequency    = 0;
        e.familiarity  = 0.f;
        words_[word]   = std::move(e);
        vocab_.push_back(word);
    }

    // Encode: word → concept vector
    // Dynamically auto-assigns a vector if the word is unknown.
    std::vector<float> encode(const std::string& word) {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = words_.find(word);
        if (it == words_.end()) {
            WordEntry e;
            std::mt19937 rng(std::hash<std::string>{}(word));
            e.vec = random_vec(rng);
            e.frequency    = 0;
            e.familiarity  = 0.f;
            words_[word]   = e;
            vocab_.push_back(word);
            return e.vec;
        }
        return it->second.vec;
    }

    // Decode: concept vector → top-k words with similarity scores
    std::vector<std::pair<std::string, float>>
    decode(const std::vector<float>& concept_vec, int k = 5) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<std::pair<float, std::string>> sims;
        sims.reserve(words_.size());
        for (const auto& [w, e] : words_)
            sims.push_back({cosine(e.vec, concept_vec), w});

        int kk = std::min(k, (int)sims.size());
        std::partial_sort(sims.begin(), sims.begin() + kk, sims.end(),
            [](const auto& a, const auto& b){ return a.first > b.first; });

        std::vector<std::pair<std::string, float>> result;
        result.reserve(kk);
        for (int i = 0; i < kk; i++)
            result.push_back({sims[i].second, sims[i].first});
        return result;
    }

    // Best single word for a concept vector
    std::string best_word(const std::vector<float>& concept_vec) const {
        auto top = decode(concept_vec, 1);
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
            it = words_.find(word);
        }
        auto& e = it->second;
        e.frequency++;
        e.familiarity = 0.9f * e.familiarity + 0.1f;  // smoothed

        // Move embedding toward current activation (Hebbian)
        if ((int)som_activation.size() == n_dims) {
            for (int i = 0; i < n_dims; i++)
                e.vec[i] += lr * (som_activation[i] - e.vec[i]);
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
        int n = (int)vocab_.size();
        f.write((const char*)&n, sizeof(int));
        for (const auto& w : vocab_) {
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
        }
        return l;
    }
};

} // namespace brain2
