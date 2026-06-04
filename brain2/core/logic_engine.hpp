#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "core/basal_ganglia.hpp"
#include "core/language.hpp"
#include "core/symbolic.hpp"
#include "core/binding_memory.hpp"
#include "core/episodic.hpp"
#include "core/som.hpp"
#include "core/working_mem.hpp"
#include "core/analogy.hpp"
#include "core/memoization.hpp"
#include "core/predictive_coding.hpp"

namespace brain2 {

class LogicEngine {
public:
    int n_dims;
    Language& language;
    Symbolic& symbolic;
    BindingMemory& binding;
    EpisodicMemory& episodic;
    SOM& som;
    WorkingMemory& working_mem;
    AnalogyEngine& analogy;
    MemoizationCache& memo_cache;
    PredictiveCodingLayer& pc_wm;
    std::vector<std::string>& spoken_words;

    LogicEngine(int n, Language& lang, Symbolic& sym, BindingMemory& bind, 
                EpisodicMemory& epi, SOM& som_ref, WorkingMemory& wm, 
                AnalogyEngine& ana, MemoizationCache& cache, PredictiveCodingLayer& pc, std::vector<std::string>& spoken)
        : n_dims(n), language(lang), symbolic(sym), binding(bind), episodic(epi), 
          som(som_ref), working_mem(wm), analogy(ana), memo_cache(cache), pc_wm(pc), spoken_words(spoken) {}

    // Cosine similarity helper
    float cosine(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty()) return 0.f;
        float dot = 0.f, na = 0.f, nb = 0.f;
        size_t lim = std::min(a.size(), b.size());
        for (size_t i = 0; i < lim; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-8f || nb < 1e-8f) return 0.f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    std::vector<float> execute_op(Op op, Scratchpad& pad, bool commit = true) {
        // Fast path: Memoization Check (only for deterministic mathematical ops and complex binds)
        if (op == Op::MATH_DIV || op == Op::MATH_SUB || op == Op::MATH_ADD || op == Op::MATH_MUL || op == Op::MATH_POW || op == Op::BIND_QUERY) {
            std::string cache_key = "OP_" + std::to_string((int)op) + "_";
            auto subj = pad.read("subject");
            auto obj = pad.read("object");
            auto rel = pad.read("relation");
            if (!subj.empty()) cache_key += language.best_word(subj) + "_";
            if (!obj.empty()) cache_key += language.best_word(obj) + "_";
            if (!rel.empty()) cache_key += language.best_word(rel) + "_";
            
            if (memo_cache.has_vec(cache_key)) {
                auto cached_res = memo_cache.get_vec(cache_key);
                pad.write("result", cached_res, "memo");
                return compute_context(pad);
            }
        }

        switch (op) {
            case Op::READ: {
                auto res = pad.read("result");
                pad.write("focus", res, "attn");
                break;
            }
            case Op::WRITE: {
                auto focus = pad.read("focus");
                if (!focus.empty()) pad.write("subject", focus, "mem");
                break;
            }
            case Op::MATH_SUB: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::stoi(s_sym) - std::stoi(o_sym);
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            auto enc = language.encode(res_sym);
                            pad.write("result", enc, "math");
                            std::string key = "OP_" + std::to_string((int)op) + "_" + s_sym + "_" + o_sym + "__";
                            memo_cache.put_vec(key, enc);
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_DIV: {
                auto res = pad.read("result");
                auto rel = pad.read("relation");
                if (!res.empty() && !rel.empty()) {
                    auto r_sym = language.best_word(res);
                    auto d_sym = language.best_word(rel);
                    if (!r_sym.empty() && !d_sym.empty()) {
                        try {
                            int d_val = std::stoi(d_sym);
                            if (d_val != 0) {
                                float f_q = std::stoi(r_sym) / (float)d_val;
                                int q = (int)std::floor(f_q);
                                std::string fin_sym = std::to_string(q);
                                if (!language.knows(fin_sym)) {
                                    language.register_word(fin_sym);
                                    symbolic.bind(fin_sym);
                                }
                                auto enc = language.encode(fin_sym);
                                pad.write("result", enc, "math");
                                std::string key = "OP_" + std::to_string((int)op) + "__" + r_sym + "_" + d_sym + "_";
                                memo_cache.put_vec(key, enc);
                            }
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::STORE_TMP: {
                auto res = pad.read("result");
                if (!res.empty()) {
                    pad.write("relation", res, "math");
                }
                break;
            }
            case Op::MATH_ADD: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::stoi(s_sym) + std::stoi(o_sym);
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            auto enc = language.encode(res_sym);
                            pad.write("result", enc, "math");
                            std::string key = "OP_" + std::to_string((int)op) + "_" + s_sym + "_" + o_sym + "__";
                            memo_cache.put_vec(key, enc);
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_MUL: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::stoi(s_sym) * std::stoi(o_sym);
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            auto enc = language.encode(res_sym);
                            pad.write("result", enc, "math");
                            std::string key = "OP_" + std::to_string((int)op) + "_" + s_sym + "_" + o_sym + "__";
                            memo_cache.put_vec(key, enc);
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_POW: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            int res = std::pow(std::stoi(s_sym), std::stoi(o_sym));
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            auto enc = language.encode(res_sym);
                            pad.write("result", enc, "math");
                            std::string key = "OP_" + std::to_string((int)op) + "_" + s_sym + "_" + o_sym + "__";
                            memo_cache.put_vec(key, enc);
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_FACT: {
                auto subj = pad.read("subject");
                if (!subj.empty()) {
                    auto s_sym = language.best_word(subj);
                    if (!s_sym.empty()) {
                        try {
                            int n = std::stoi(s_sym);
                            long long res = 1;
                            for (int i = 2; i <= n; i++) res *= i;
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("result", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_FACT_REL: {
                auto rel = pad.read("relation");
                if (!rel.empty()) {
                    auto r_sym = language.best_word(rel);
                    if (!r_sym.empty()) {
                        try {
                            int n = std::stoi(r_sym);
                            long long res = 1;
                            for (int i = 2; i <= n; i++) res *= i;
                            std::string res_sym = std::to_string(res);
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym);
                            }
                            pad.write("relation", language.encode(res_sym), "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_DIV_FLOAT: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj);
                    auto o_sym = language.best_word(obj);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            float denom = std::stof(o_sym);
                            if (std::abs(denom) > 1e-6f) {
                                double val_d = std::stod(s_sym) / std::stod(o_sym);
                                // Use round-half-to-even (banker's rounding) to match Python's round()
                                double rounded = std::rint(val_d * 100.0) / 100.0;
                                char buf[32];
                                std::snprintf(buf, sizeof(buf), "%.2f", rounded);
                                std::string res_sym(buf);
                                if (!language.knows(res_sym)) {
                                    language.register_word(res_sym);
                                    symbolic.bind(res_sym);
                                }
                                pad.write("result", language.encode(res_sym), "math");
                            }
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::COMPARE: {
                auto result = pad.read("result");
                auto obj    = pad.read("object");
                float sim = 0.f;
                if (!result.empty() && !obj.empty()) {
                    sim = cosine(result, obj);
                }
                std::vector<float> sim_vec(n_dims, sim);
                pad.write("comparison", sim_vec, "eval");
                break;
            }
            case Op::NOT: {
                auto obj = pad.read("object");
                if (!obj.empty()) {
                    for (auto& x : obj) x = -x;
                    pad.write("object", obj, "modifier");
                }
                break;
            }
            case Op::BIND_QUERY: {
                auto subj = pad.read("subject");
                auto rel  = pad.read("relation");
                if (!subj.empty() && !rel.empty()) {
                    auto [ans, conf] = binding.query(subj, rel, true, 0.3f, 4);
                    // Always write confidence so callers can gate on it
                    pad.write("confidence", std::vector<float>{conf}, "query");
                    if (conf >= 0.25f) {
                        // Known answer — write result
                        pad.write("result", ans, "binding");
                        std::string s_sym = language.best_word(subj);
                        std::string r_sym = language.best_word(rel);
                        std::string key = "OP_" + std::to_string((int)op) + "_" + s_sym + "__" + r_sym + "_";
                        memo_cache.put_vec(key, ans);
                    } else {
                        // Unknown — write zero vector ("I don't know")
                        pad.write("result", std::vector<float>(n_dims, 0.f), "query");
                    }
                }
                break;
            }
            case Op::BIND_ISA: {
                auto subj = pad.read("subject");
                if (!subj.empty()) {
                    auto isa_vec = language.encode("isa");
                    auto [ans, conf] = binding.query(subj, isa_vec, true);
                    if (conf > 0.5f) {
                        pad.write("result", ans, "binding");
                    }
                }
                break;
            }
            case Op::RETRIEVE: {
                auto focus = pad.read("focus");
                if (!focus.empty()) {
                    auto topk = episodic.retrieve_topk(focus, 1);
                    if (!topk.empty()) {
                        auto* ep = episodic.get_episode(topk[0].second);
                        if (ep) {
                            if (!ep->payload.empty()) {
                                pad.write("result", ep->payload, "episodic");
                            } else if (!ep->root.summary_spike.empty()) {
                                std::vector<float> dense(n_dims, 0.f);
                                size_t lim = std::min(dense.size(), ep->root.summary_spike.size());
                                for (size_t i = 0; i < lim; i++) {
                                    if (ep->root.summary_spike[i]) dense[i] = 1.0f;
                                }
                                pad.write("result", dense, "episodic");
                            }
                        }
                    }
                }
                break;
            }
            case Op::ANALOGY: {
                auto a   = pad.read("subject");
                auto rel = pad.read("relation");
                auto ctx = working_mem.context();
                if (!a.empty() && !rel.empty()) {
                    auto mapped = analogy.structure_map(a, rel, ctx.empty() ? std::vector<float>(n_dims, 0.f) : ctx);
                    pad.write("result", mapped, "analogy");
                }
                break;
            }
            case Op::ASK_USER: {
                auto ask_vec = language.encode("ask");
                pad.write("result", ask_vec, "curiosity");
                break;
            }
            case Op::STORE_SUBJ: {
                auto sens = pad.read("sensory_input");
                if (!sens.empty()) pad.write("subject", sens, "context");
                break;
            }
            case Op::STORE_REL: {
                auto sens = pad.read("sensory_input");
                if (!sens.empty()) pad.write("relation", sens, "context");
                break;
            }
            case Op::STORE_OBJ: {
                auto sens = pad.read("sensory_input");
                if (!sens.empty()) pad.write("object", sens, "context");
                break;
            }
            case Op::SPEAK: {
                auto res = pad.read("result");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_SUBJ: {
                auto res = pad.read("subject");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_REL: {
                auto res = pad.read("relation");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_OBJ: {
                auto res = pad.read("object");
                if (!res.empty()) {
                    std::string word = language.best_word(res);
                    if (!word.empty() && commit) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::ATTEND: {
                auto subj = pad.read("subject");
                auto rel = pad.read("relation");
                auto obj = pad.read("object");
                
                auto q_vec = language.encode("?");
                
                if (!subj.empty() && cosine(subj, q_vec) > 0.8f) {
                    pad.write("focus", subj, "attention");
                } else if (!rel.empty() && cosine(rel, q_vec) > 0.8f) {
                    pad.write("focus", rel, "attention");
                } else if (!obj.empty() && cosine(obj, q_vec) > 0.8f) {
                    pad.write("focus", obj, "attention");
                }
                break;
            }
            case Op::PREDICT_WM: {
                if (!pc_wm.prediction.empty()) {
                    pad.write("result", pc_wm.prediction, "prediction");
                }
                break;
            }
            case Op::CHAIN_FOLLOW: {
                // Iterative multi-hop traversal along a relation (e.g. "causes").
                // Avoids O(n^depth) blowup of recursive query by using BFS one step at a time.
                // Max 10 hops; stops on low confidence or cycle.
                auto current = pad.read("subject");
                auto rel     = pad.read("relation");
                if (!current.empty() && !rel.empty()) {
                    auto start   = current;
                    float threshold = 0.3f;
                    int   max_hops  = 10;
                    float best_conf = 0.f;
                    for (int hop = 0; hop < max_hops; hop++) {
                        auto [next, conf] = binding.query(current, rel, true, threshold);
                        if (conf < threshold || next.empty()) break;
                        // Cycle guard: stop if next is similar to start
                        if (cosine(next, start) > 0.92f) break;
                        current   = next;
                        best_conf = conf;
                    }
                    pad.write("result",     current,                       "chain");
                    pad.write("confidence", std::vector<float>{best_conf}, "chain");
                }
                break;
            }
            case Op::HALT:
            default:
                break;
        }

        return compute_context(pad);
    }

private:
    std::vector<float> compute_context(Scratchpad& pad) {
        std::vector<float> ctx(n_dims, 0.f);
        auto slots = pad.slot_names();
        for (const auto& s : slots) {
            auto val = pad.read(s);
            
            std::string slot_sym = "SLOT_" + s;
            if (!symbolic.knows(slot_sym)) symbolic.bind(slot_sym);
            auto slot_embed = symbolic.lookup(slot_sym);
            
            size_t lim = std::min(val.size(), ctx.size());
            for (size_t i = 0; i < lim; i++) {
                ctx[i] += val[i] * slot_embed[i];
            }
        }
        float norm_val = 0.f;
        for (float x : ctx) norm_val += x * x;
        if (norm_val > 1e-8f) {
            norm_val = std::sqrt(norm_val);
            for (float& x : ctx) x /= norm_val;
        }
        return ctx;
    }

public:
    void expand_dims(int new_dims) {
        if (new_dims <= n_dims) return;
        n_dims = new_dims;
    }
};

}
