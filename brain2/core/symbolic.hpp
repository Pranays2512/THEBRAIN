#pragma once
/*
 * symbolic.hpp — Symbolic Binding, Component 10 of Brain v2
 *
 * Math/logic symbols need stable concept vectors — unlike natural language
 * words that drift via Hebbian learning, symbolic bindings are fixed once set.
 *
 * Symbolic binding table: symbol string → stable concept vector
 * The concept vector lives in the same space as SOM activations,
 * so math operations produce SOM-compatible outputs.
 *
 * Symbolic operation bindings:
 *   Each symbol can have an "operator function" that takes two concept vectors
 *   and produces a result concept vector. For math:
 *     "+" operator: blend_add(a, b)    — sum-normalized
 *     "-" operator: blend_sub(a, b)    — difference-normalized
 *     "=" operator: similarity check   — returns identity if similar
 *     ">" operator: compare magnitude
 *
 * This is NOT a symbolic AI system. Operators here produce concept vectors,
 * not discrete boolean results. The "reasoning" emerges from the Predictor
 * learning sequences of (concept_a, operator, concept_b) → concept_result.
 *
 * Grounding: symbol vectors should be seeded with a distinct random pattern
 * unique to each symbol, then left stable. The brain learns what "+" means
 * by observing many (a + b = a+b) sequences — not from hardcoded rules.
 */

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <random>
#include <functional>

namespace brain2 {

enum class SymbolOp {
    NONE,       // No operation (pure symbol)
    ADD,        // Blend sum
    SUBTRACT,   // Blend difference
    MULTIPLY,   // Element-wise product (normalized)
    DIVIDE,     // Element-wise division (normalized)
    COMPARE,    // Similarity measure → scaled identity
    SEQUENCE,   // Sequence continuation hint
    NEGATE,     // Flip sign
};

struct SymbolEntry {
    std::vector<float> vec;  // stable concept vector in SOM space
    SymbolOp           op;   // what operation this symbol implies
    std::string        category; // "math", "logic", "relation", "unit", etc.
    int                use_count;
};

class Symbolic {
public:
    int n_dims;

private:
    std::unordered_map<std::string, SymbolEntry> table_;
    std::unique_ptr<std::mutex>                  mtx_;

    static float dot(const std::vector<float>& a,
                     const std::vector<float>& b) noexcept {
        float s = 0.f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    static float norm(const std::vector<float>& v) noexcept {
        return std::sqrt(dot(v, v));
    }

    static std::vector<float> normalize(std::vector<float> v) {
        float n = norm(v);
        if (n < 1e-8f) return v;
        for (auto& x : v) x /= n;
        return v;
    }

    // Deterministic unique vector from symbol string
    std::vector<float> seed_vec(const std::string& sym) const {
        std::mt19937 rng(std::hash<std::string>{}(sym) ^ 0xDEADBEEF);
        std::normal_distribution<float> dist(0.f, 1.f);
        std::vector<float> v(n_dims);
        for (auto& x : v) x = dist(rng);
        return normalize(v);
    }

public:
    Symbolic() : n_dims(0), mtx_(std::make_unique<std::mutex>()) {}

    Symbolic(int n_dims) : n_dims(n_dims),
                           mtx_(std::make_unique<std::mutex>()) {}

    Symbolic(Symbolic&&)            = default;
    Symbolic& operator=(Symbolic&&) = default;
    Symbolic(const Symbolic&)       = delete;
    Symbolic& operator=(const Symbolic&) = delete;

    // Register a symbol with a fixed concept vector
    // If vec empty: auto-seed from symbol string (deterministic)
    void bind(const std::string& symbol,
              const std::vector<float>& vec = {},
              SymbolOp op = SymbolOp::NONE,
              const std::string& category = "") {
        std::lock_guard<std::mutex> lock(*mtx_);
        if (table_.count(symbol)) return;  // stable — don't overwrite
        SymbolEntry e;
        e.vec = ((int)vec.size() == n_dims) ? vec : seed_vec(symbol);
        e.op         = op;
        e.category   = category;
        e.use_count  = 0;
        table_[symbol] = std::move(e);
    }

    // Lookup symbol → concept vector (zero if unknown)
    std::vector<float> lookup(const std::string& symbol) {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = table_.find(symbol);
        if (it == table_.end()) return std::vector<float>(n_dims, 0.f);
        it->second.use_count++;
        return it->second.vec;
    }

    // Apply symbolic operation: op_symbol(a, b) → result concept vector
    // Returns zero vector if symbol has no op
    std::vector<float> apply(const std::string& op_symbol,
                              const std::vector<float>& a,
                              const std::vector<float>& b) {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto it = table_.find(op_symbol);
        if (it == table_.end()) return std::vector<float>(n_dims, 0.f);
        it->second.use_count++;

        int n = n_dims;
        std::vector<float> result(n, 0.f);

        switch (it->second.op) {
            case SymbolOp::ADD: {
                for (int i = 0; i < n; i++) result[i] = a[i] + b[i];
                return normalize(result);
            }
            case SymbolOp::SUBTRACT: {
                for (int i = 0; i < n; i++) result[i] = a[i] - b[i];
                return normalize(result);
            }
            case SymbolOp::MULTIPLY: {
                for (int i = 0; i < n; i++) result[i] = a[i] * b[i];
                return normalize(result);
            }
            case SymbolOp::DIVIDE: {
                for (int i = 0; i < n; i++) {
                    // Prevent division by zero with small epsilon
                    float denom = (std::abs(b[i]) < 1e-6f) ? 1e-6f * (b[i] >= 0 ? 1 : -1) : b[i];
                    result[i] = a[i] / denom;
                }
                return normalize(result);
            }
            case SymbolOp::COMPARE: {
                // Return similarity score as uniform scaled identity vector
                float na = norm(a), nb = norm(b);
                float sim = (na > 1e-8f && nb > 1e-8f)
                    ? std::max(0.f, dot(a,b) / (na * nb))
                    : 0.f;
                for (int i = 0; i < n; i++) result[i] = sim;
                return result;
            }
            case SymbolOp::NEGATE: {
                for (int i = 0; i < n; i++) result[i] = -a[i];
                return result;
            }
            case SymbolOp::SEQUENCE: {
                // Hint: b is the expected continuation after a
                // Return blend slightly toward b
                for (int i = 0; i < n; i++) result[i] = 0.3f*a[i] + 0.7f*b[i];
                return normalize(result);
            }
            default:
                return result;
        }
    }

    // Reverse lookup: closest symbol to a concept vector
    std::string nearest_symbol(const std::vector<float>& vec) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::string best_sym;
        float best_sim = -2.f;
        float vn = norm(vec);
        if (vn < 1e-8f) return "";
        for (const auto& [sym, e] : table_) {
            float en = norm(e.vec);
            if (en < 1e-8f) continue;
            float sim = dot(vec, e.vec) / (vn * en);
            if (sim > best_sim) { best_sim = sim; best_sym = sym; }
        }
        return best_sym;
    }

    bool knows(const std::string& sym) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return table_.count(sym) > 0;
    }

    int symbol_count() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        return (int)table_.size();
    }

    std::vector<std::string> symbols() const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::vector<std::string> out;
        out.reserve(table_.size());
        for (const auto& [k, _] : table_) out.push_back(k);
        return out;
    }

    // Seed standard math/logic symbols (useful for initial binding)
    void seed_math_symbols() {
        bind("+",  {}, SymbolOp::ADD,      "math");
        bind("-",  {}, SymbolOp::SUBTRACT, "math");
        bind("*",  {}, SymbolOp::MULTIPLY, "math");
        bind("/",  {}, SymbolOp::DIVIDE,   "math");
        bind("=",  {}, SymbolOp::COMPARE,  "relation");
        bind(">",  {}, SymbolOp::COMPARE,  "relation");
        bind("<",  {}, SymbolOp::COMPARE,  "relation");
        bind("->", {}, SymbolOp::SEQUENCE, "logic");
        bind("!",  {}, SymbolOp::NEGATE,   "logic");
        bind("0",  {}, SymbolOp::NONE,     "number");
        bind("1",  {}, SymbolOp::NONE,     "number");
        bind("2",  {}, SymbolOp::NONE,     "number");
        bind("3",  {}, SymbolOp::NONE,     "number");
        bind("pi", {}, SymbolOp::NONE,     "constant");
        bind("e",  {}, SymbolOp::NONE,     "constant");
    }

    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(*mtx_);
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Symbolic::save: cannot open " + path);
        f.write((const char*)&n_dims, sizeof(int));
        int n = (int)table_.size();
        f.write((const char*)&n, sizeof(int));
        for (const auto& [sym, e] : table_) {
            int slen = (int)sym.size();
            f.write((const char*)&slen, sizeof(int));
            f.write(sym.data(), slen);
            f.write((const char*)e.vec.data(),
                    (std::streamsize)(n_dims * sizeof(float)));
            int op = (int)e.op;
            f.write((const char*)&op, sizeof(int));
            int clen = (int)e.category.size();
            f.write((const char*)&clen, sizeof(int));
            f.write(e.category.data(), clen);
            f.write((const char*)&e.use_count, sizeof(int));
        }
    }

    static Symbolic load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Symbolic::load: cannot open " + path);
        Symbolic s;
        f.read((char*)&s.n_dims, sizeof(int));
        int n; f.read((char*)&n, sizeof(int));
        for (int i = 0; i < n; i++) {
            int slen; f.read((char*)&slen, sizeof(int));
            std::string sym(slen, '\0');
            f.read(sym.data(), slen);
            SymbolEntry e;
            e.vec.resize(s.n_dims);
            f.read((char*)e.vec.data(),
                   (std::streamsize)(s.n_dims * sizeof(float)));
            int op; f.read((char*)&op, sizeof(int));
            e.op = (SymbolOp)op;
            int clen; f.read((char*)&clen, sizeof(int));
            e.category.resize(clen);
            f.read(e.category.data(), clen);
            f.read((char*)&e.use_count, sizeof(int));
            s.table_[sym] = std::move(e);
        }
        s.mtx_ = std::make_unique<std::mutex>();
        return s;
    }
};

} // namespace brain2
