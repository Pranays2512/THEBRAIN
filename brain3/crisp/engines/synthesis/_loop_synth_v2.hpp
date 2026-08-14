#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

namespace brain3 { 
namespace engines { 
namespace synthesis {

struct LoopSpecV2 {
    std::string kind;
    
    // cond params
    int init;
    std::string lo, hi, cond, upd;
    std::function<int(int)> r_lo;
    std::function<int(int)> r_hi;
    std::function<bool(int, int)> cf;
    std::function<int(int, int)> uf;

    // two-state params
    int ia, ib;
    std::string na, nb, ret;
    std::function<int(int, int, int)> naf;
    std::function<int(int, int, int)> nbf;
};

class LoopSynthV2 {
private:
    struct RangeDef {
        std::string lo_str, hi_str;
        std::function<int(int)> r_lo;
        std::function<int(int)> r_hi;
    };
    
    std::vector<RangeDef> RANGES = {
        {"0", "n",     [](int n){ return 0; }, [](int n){ return n; }},
        {"1", "n + 1", [](int n){ return 1; }, [](int n){ return n + 1; }},
        {"1", "n",     [](int n){ return 1; }, [](int n){ return n; }},
        {"2", "n + 1", [](int n){ return 2; }, [](int n){ return n + 1; }}
    };

    std::map<std::string, std::function<int(int, int, int)>> STATE = {
        {"a", [](int a, int b, int i){ return a; }},
        {"b", [](int a, int b, int i){ return b; }},
        {"a + b", [](int a, int b, int i){ return a + b; }},
        {"a + i", [](int a, int b, int i){ return a + i; }},
        {"b + i", [](int a, int b, int i){ return b + i; }},
        {"a * i", [](int a, int b, int i){ return a * i; }},
        {"a + 1", [](int a, int b, int i){ return a + 1; }}
    };

    std::vector<std::pair<int, int>> INITS2 = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};

    std::map<std::string, std::function<bool(int, int)>> CONDS = {
        {"n % i == 0", [](int i, int n){ return i != 0 && n % i == 0; }},
        {"i % 2 == 0", [](int i, int n){ return i % 2 == 0; }},
        {"i % 2 == 1", [](int i, int n){ return i % 2 == 1; }},
        {"True",       [](int i, int n){ return true; }},
        {"i % 2 != 0", [](int i, int n){ return i % 2 != 0; }}
    };

    std::map<std::string, std::function<int(int, int)>> UPD = {
        {"acc + 1", [](int acc, int i){ return acc + 1; }},
        {"acc + i", [](int acc, int i){ return acc + i; }},
        {"acc * i", [](int acc, int i){ return acc * i; }}
    };

public:
    LoopSpecV2* synth_two(const std::vector<std::pair<int, int>>& examples) {
        int cut = std::max(3, (int)(examples.size() * 0.6));
        
        for (auto [ia, ib] : INITS2) {
            for (const auto& r : RANGES) {
                for (const auto& [na, naf] : STATE) {
                    for (const auto& [nb, nbf] : STATE) {
                        for (std::string ret : {"a", "b"}) {
                            
                            auto f = [&](int n) {
                                int a = ia, b = ib;
                                for (int i = r.r_lo(n); i < r.r_hi(n); i++) {
                                    int next_a = naf(a, b, i);
                                    int next_b = nbf(a, b, i);
                                    a = next_a;
                                    b = next_b;
                                }
                                return (ret == "a") ? a : b;
                            };

                            bool ok = true;
                            for (const auto& ex : examples) {
                                if (f(ex.first) != ex.second) { ok = false; break; }
                            }
                            if (ok) {
                                auto spec = new LoopSpecV2();
                                spec->kind = "two";
                                spec->ia = ia; spec->ib = ib;
                                spec->lo = r.lo_str; spec->hi = r.hi_str;
                                spec->na = na; spec->nb = nb; spec->ret = ret;
                                return spec;
                            }
                        }
                    }
                }
            }
        }
        return nullptr;
    }

    LoopSpecV2* synth_cond(const std::vector<std::pair<int, int>>& examples) {
        for (int init : {0, 1}) {
            for (const auto& r : RANGES) {
                for (const auto& [cc, cf] : CONDS) {
                    for (const auto& [uc, uf] : UPD) {
                        
                        auto f = [&](int n) {
                            int acc = init;
                            for (int i = r.r_lo(n); i < r.r_hi(n); i++) {
                                if (cf(i, n)) {
                                    acc = uf(acc, i);
                                }
                            }
                            return acc;
                        };

                        bool ok = true;
                        for (const auto& ex : examples) {
                            if (f(ex.first) != ex.second) { ok = false; break; }
                        }
                        if (ok) {
                            auto spec = new LoopSpecV2();
                            spec->kind = "cond";
                            spec->init = init;
                            spec->lo = r.lo_str; spec->hi = r.hi_str;
                            spec->cond = cc; spec->upd = uc;
                            return spec;
                        }
                    }
                }
            }
        }
        return nullptr;
    }

    LoopSpecV2* synthesize(const std::vector<std::pair<int, int>>& examples) {
        if (examples.size() < 4) return nullptr;
        if (auto p = synth_cond(examples)) return p;
        if (auto p = synth_two(examples)) return p;
        return nullptr;
    }

    std::string render(const std::string& fn, const LoopSpecV2& s) {
        if (s.kind == "two") {
            return "int " + fn + "(int n) {\n"
                   "    int a = " + std::to_string(s.ia) + ", b = " + std::to_string(s.ib) + ";\n"
                   "    for (int i = " + s.lo + "; i < " + s.hi + "; i++) {\n"
                   "        int next_a = " + s.na + ";\n"
                   "        int next_b = " + s.nb + ";\n"
                   "        a = next_a; b = next_b;\n"
                   "    }\n"
                   "    return " + s.ret + ";\n"
                   "}\n";
        } else {
            return "int " + fn + "(int n) {\n"
                   "    int acc = " + std::to_string(s.init) + ";\n"
                   "    for (int i = " + s.lo + "; i < " + s.hi + "; i++) {\n"
                   "        if (" + s.cond + ") {\n"
                   "            acc = " + s.upd + ";\n"
                   "        }\n"
                   "    }\n"
                   "    return acc;\n"
                   "}\n";
        }
    }
};

}}}
