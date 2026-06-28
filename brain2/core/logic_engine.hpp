#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "debug.hpp"
#include "core/basal_ganglia.hpp"
#include "core/language.hpp"
#include "core/symbolic.hpp"
#include "core/binding_memory.hpp"
#include "core/episodic.hpp"
#include "core/som.hpp"
#include "core/working_mem.hpp"
#include "core/analogy.hpp"
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
    PredictiveCodingLayer& pc_wm;
    std::vector<std::string>& spoken_words;

    LogicEngine(int n, Language& lang, Symbolic& sym, BindingMemory& bind, 
                EpisodicMemory& epi, SOM& som_ref, WorkingMemory& wm, 
                AnalogyEngine& ana, PredictiveCodingLayer& pc, std::vector<std::string>& spoken)
        : n_dims(n), language(lang), symbolic(sym), binding(bind), episodic(epi), 
          som(som_ref), working_mem(wm), analogy(ana), pc_wm(pc), spoken_words(spoken) {}

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
                    auto s_sym = language.best_word(subj, {}, 0);
                    auto o_sym = language.best_word(obj, {}, 0);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            float res = std::stof(s_sym) - std::stof(o_sym);
                            
                            // Format cleanly
                            char buf[32];
                            snprintf(buf, sizeof(buf), "%g", res);
                            std::string res_sym(buf);
                            
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym, {}, SymbolOp::NONE, "number");
                            }
                            auto enc = language.encode(res_sym);
                            pad.write("result", enc, "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_DIV: {
                auto res = pad.read("result");
                auto rel = pad.read("relation");
                if (!res.empty() && !rel.empty()) {
                    auto r_sym = language.best_word(res, {}, 0);
                    auto d_sym = language.best_word(rel, {}, 0);
                    if (!r_sym.empty() && !d_sym.empty()) {
                        try {
                            float d_val = std::stof(d_sym);
                            if (std::abs(d_val) > 1e-6f) {
                                float f_q = std::stof(r_sym) / d_val;
                                
                                char buf[32];
                                snprintf(buf, sizeof(buf), "%g", f_q);
                                std::string fin_sym(buf);
                                
                                if (!language.knows(fin_sym)) {
                                    language.register_word(fin_sym);
                                    symbolic.bind(fin_sym, {}, SymbolOp::NONE, "number");
                                }
                                auto enc = language.encode(fin_sym);
                                pad.write("result", enc, "math");
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
                    auto s_sym = language.best_word(subj, {}, 0);
                    auto o_sym = language.best_word(obj, {}, 0);
                    if (!s_sym.empty() && !o_sym.empty()) {
                        try {
                            float res = std::stof(s_sym) + std::stof(o_sym);
                            
                            char buf[32];
                            snprintf(buf, sizeof(buf), "%g", res);
                            std::string res_sym(buf);
                            
                            if (!language.knows(res_sym)) {
                                language.register_word(res_sym);
                                symbolic.bind(res_sym, {}, SymbolOp::NONE, "number");
                            }
                            auto enc = language.encode(res_sym);
                            pad.write("result", enc, "math");
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_MUL: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj, {}, 0);
                    auto o_sym = language.best_word(obj, {}, 0);
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
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_POW: {
                auto subj = pad.read("subject");
                auto obj  = pad.read("object");
                if (!subj.empty() && !obj.empty()) {
                    auto s_sym = language.best_word(subj, {}, 0);
                    auto o_sym = language.best_word(obj, {}, 0);
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
                        } catch (...) {}
                    }
                }
                break;
            }
            case Op::MATH_FACT: {
                auto subj = pad.read("subject");
                if (!subj.empty()) {
                    auto s_sym = language.best_word(subj, {}, 0);
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
                    auto r_sym = language.best_word(rel, {}, 0);
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
                    auto s_sym = language.best_word(subj, {}, 0);
                    auto o_sym = language.best_word(obj, {}, 0);
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
            case Op::MATH_POLY: {
                auto x_v = pad.read("subject");
                auto a_v = pad.read("object");
                auto b_v = pad.read("a_operator");
                auto c_v = pad.read("focus");
                if (!x_v.empty() && !a_v.empty() && !b_v.empty() && !c_v.empty()) {
                    auto x_sym = language.best_word(x_v, {}, 0);
                    auto a_sym = language.best_word(a_v, {}, 0);
                    auto b_sym = language.best_word(b_v, {}, 0);
                    auto c_sym = language.best_word(c_v, {}, 0);
                    if (!x_sym.empty() && !a_sym.empty() && !b_sym.empty() && !c_sym.empty()) {
                        try {
                            int x = std::stoi(x_sym);
                            int a = std::stoi(a_sym);
                            int b = std::stoi(b_sym);
                            int c = std::stoi(c_sym);
                            int res = a * x * x + b * x + c;
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
            case Op::MATH_QUAD: {
                auto b_v = pad.read("object");
                auto c_v = pad.read("a_operator");
                if (!b_v.empty() && !c_v.empty()) {
                    auto b_sym = language.best_word(b_v, {}, 0);
                    auto c_sym = language.best_word(c_v, {}, 0);
                    if (!b_sym.empty() && !c_sym.empty()) {
                        try {
                            int b = std::stoi(b_sym);
                            int c = std::stoi(c_sym);
                            int delta = b*b - 4*c;
                            if (delta >= 0) {
                                int r1 = (-b + std::sqrt(delta)) / 2;
                                int r2 = (-b - std::sqrt(delta)) / 2;
                                if (r1 > r2) std::swap(r1, r2);
                                std::string res_sym = std::to_string(r1) + "_and_" + std::to_string(r2);
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
                        std::string s_sym = language.best_word(subj, {}, 0);
                        std::string r_sym = language.best_word(rel, {}, 0);
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
                // Query Episodic Memory using the exact centroid sequence it expects
                auto ctx_map = pad.read("context_map");
                if (::b2::debug_on()) fprintf(stderr, "DEBUG: Executing OP_RETRIEVE. ctx_map.empty() = %d\\n", ctx_map.empty());
                if (!ctx_map.empty()) {
                    // Search Episodic Memory using the full continuous 65536D context
                    auto topk = episodic.retrieve_topk(ctx_map, 1);
                    if (::b2::debug_on()) fprintf(stderr, "DEBUG: OP_RETRIEVE topk.empty() = %d\\n", topk.empty());
                    if (!topk.empty()) {
                        if (::b2::debug_on()) fprintf(stderr, "DEBUG: OP_RETRIEVE topk[0].first = %f\\n", topk[0].first);
                        if (topk[0].first > 0.001f) { // Lowered threshold for Continuous-Discrete Dot Product
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
                }
                break;
            }
            case Op::ANALOGY: {
                auto a   = pad.read("subject");
                auto rel = pad.read("relation");
                if (!a.empty() && !rel.empty()) {
                    auto ctx = std::vector<float>(n_dims, 0.f); 
                    auto mapped = analogy.structure_map(a, rel, ctx);
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
                if (!res.empty() && commit) {
                    std::string word = language.best_word(res);
                    if (!word.empty()) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_SUBJ: {
                auto res = pad.read("subject");
                if (!res.empty() && commit) {
                    std::string word = language.best_word(res);
                    if (!word.empty()) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_REL: {
                auto res = pad.read("relation");
                if (!res.empty() && commit) {
                    std::string word = language.best_word(res);
                    if (!word.empty()) {
                        spoken_words.push_back(word);
                    }
                }
                break;
            }
            case Op::SPEAK_OBJ: {
                auto res = pad.read("object");
                if (!res.empty() && commit) {
                    std::string word = language.best_word(res);
                    if (!word.empty()) {
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
                    int bmu = 0; float max_val = -1.f;
                    for (int i=0; i<(int)pc_wm.prediction.size(); i++) {
                        if (pc_wm.prediction[i] > max_val) { max_val = pc_wm.prediction[i]; bmu = i; }
                    }
                    pad.write("result", som.neuron_weights(bmu), "prediction");
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
