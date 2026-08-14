#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

namespace brain3 { 
namespace engines { 
namespace synthesis {

struct LoopSpecV4 {
    std::string kind;
    std::string init;
    std::string upd;
    std::string cmp;
};

class LoopSynthV4 {
private:
    std::vector<std::string> INITS = {"0", "1", "first"};

    std::map<std::string, std::function<int(int, int)>> FOLD_UPD = {
        {"acc + x",     [](int a, int x){ return a + x; }},
        {"acc * x",     [](int a, int x){ return a * x; }},
        {"std::max(acc, x)", [](int a, int x){ return std::max(a, x); }},
        {"std::min(acc, x)", [](int a, int x){ return std::min(a, x); }},
        {"acc + 1",     [](int a, int x){ return a + 1; }}
    };

    int run_fold(const std::string& init, std::function<int(int, int)> ufn, const std::vector<int>& lst) {
        if (lst.empty()) return 0;
        int acc;
        size_t start = 0;
        if (init == "first") {
            acc = lst[0];
            start = 1;
        } else {
            acc = std::stoi(init);
        }
        for (size_t i = start; i < lst.size(); i++) {
            acc = ufn(acc, lst[i]);
        }
        return acc;
    }

public:
    LoopSpecV4* synth_fold(const std::vector<std::pair<std::vector<int>, int>>& examples) {
        for (const std::string& ik : INITS) {
            for (const auto& [uc, uf] : FOLD_UPD) {
                bool ok = true;
                for (const auto& ex : examples) {
                    // special case for empty list with "first" init
                    if (ik == "first" && ex.first.empty()) continue;
                    if (run_fold(ik, uf, ex.first) != ex.second) { ok = false; break; }
                }
                if (ok) {
                    auto spec = new LoopSpecV4();
                    spec->kind = "fold";
                    spec->init = ik;
                    spec->upd = uc;
                    return spec;
                }
            }
        }
        return nullptr;
    }

    LoopSpecV4* synth_sort(const std::vector<std::pair<std::vector<int>, std::vector<int>>>& examples) {
        for (std::string cmp : {">", "<"}) {
            auto f = [&](const std::vector<int>& lst) {
                std::vector<int> a = lst;
                for (size_t i = 0; i < a.size(); i++) {
                    for (size_t j = 0; j + 1 < a.size(); j++) {
                        bool swap = (cmp == ">") ? (a[j] > a[j+1]) : (a[j] < a[j+1]);
                        if (swap) {
                            std::swap(a[j], a[j+1]);
                        }
                    }
                }
                return a;
            };
            bool ok = true;
            for (const auto& ex : examples) {
                if (f(ex.first) != ex.second) { ok = false; break; }
            }
            if (ok) {
                auto spec = new LoopSpecV4();
                spec->kind = "sort";
                spec->cmp = cmp;
                return spec;
            }
        }
        return nullptr;
    }

    LoopSpecV4* synth_member(const std::vector<std::pair<std::pair<std::vector<int>, int>, bool>>& examples) {
        auto f = [&](const std::vector<int>& lst, int t) {
            for (int x : lst) {
                if (x == t) return true;
            }
            return false;
        };
        bool ok = true;
        for (const auto& ex : examples) {
            if (f(ex.first.first, ex.first.second) != ex.second) { ok = false; break; }
        }
        if (ok) {
            auto spec = new LoopSpecV4();
            spec->kind = "member";
            return spec;
        }
        return nullptr;
    }

    LoopSpecV4* synth_nested(const std::vector<std::pair<std::vector<int>, bool>>& examples) {
        auto f = [&](const std::vector<int>& lst) {
            for (size_t i = 0; i < lst.size(); i++) {
                for (size_t j = i + 1; j < lst.size(); j++) {
                    if (lst[i] == lst[j]) return true;
                }
            }
            return false;
        };
        bool ok = true;
        for (const auto& ex : examples) {
            if (f(ex.first) != ex.second) { ok = false; break; }
        }
        if (ok) {
            auto spec = new LoopSpecV4();
            spec->kind = "nested";
            return spec;
        }
        return nullptr;
    }

    std::string render(const std::string& fn, const LoopSpecV4& s) {
        if (s.kind == "fold") {
            if (s.init == "first") {
                return "int " + fn + "(const std::vector<int>& lst) {\n"
                       "    if (lst.empty()) return 0;\n"
                       "    int acc = lst[0];\n"
                       "    for (size_t i = 1; i < lst.size(); i++) {\n"
                       "        int x = lst[i];\n"
                       "        acc = " + s.upd + ";\n"
                       "    }\n"
                       "    return acc;\n"
                       "}\n";
            }
            return "int " + fn + "(const std::vector<int>& lst) {\n"
                   "    int acc = " + s.init + ";\n"
                   "    for (int x : lst) {\n"
                   "        acc = " + s.upd + ";\n"
                   "    }\n"
                   "    return acc;\n"
                   "}\n";
        }
        if (s.kind == "member") {
            return "bool " + fn + "(const std::vector<int>& lst, int t) {\n"
                   "    for (int x : lst) {\n"
                   "        if (x == t) return true;\n"
                   "    }\n"
                   "    return false;\n"
                   "}\n";
        }
        if (s.kind == "nested") {
            return "bool " + fn + "(const std::vector<int>& lst) {\n"
                   "    for (size_t i = 0; i < lst.size(); i++) {\n"
                   "        for (size_t j = i + 1; j < lst.size(); j++) {\n"
                   "            if (lst[i] == lst[j]) return true;\n"
                   "        }\n"
                   "    }\n"
                   "    return false;\n"
                   "}\n";
        }
        if (s.kind == "sort") {
            return "std::vector<int> " + fn + "(const std::vector<int>& lst) {\n"
                   "    std::vector<int> a = lst;\n"
                   "    for (size_t i = 0; i < a.size(); i++) {\n"
                   "        for (size_t j = 0; j + 1 < a.size(); j++) {\n"
                   "            if (a[j] " + s.cmp + " a[j + 1]) {\n"
                   "                std::swap(a[j], a[j + 1]);\n"
                   "            }\n"
                   "        }\n"
                   "    }\n"
                   "    return a;\n"
                   "}\n";
        }
        return "";
    }
};

}}}
