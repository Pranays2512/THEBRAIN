#pragma once
// factorizer.hpp — native port of factorizer.eval_tree (the hot path that verifies factored
// expressions evaluate identically). Python serialises an expression tree to an S-expression
// ("(* (+ a b) c)"); C++ parses + evaluates it under a variable environment. Operators + - * /,
// leaves are numbers or variable names. Used to check meaning is preserved under factoring.
#include <cctype>
#include <map>
#include <string>
#include <vector>

namespace brain2 {

inline std::vector<std::string> _sexpr_tokens(const std::string& s) {
    std::vector<std::string> t;
    std::string cur;
    for (char c : s) {
        if (c == '(' || c == ')') {
            if (!cur.empty()) { t.push_back(cur); cur.clear(); }
            t.push_back(std::string(1, c));
        } else if (std::isspace((unsigned char)c)) {
            if (!cur.empty()) { t.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) t.push_back(cur);
    return t;
}

inline double _eval_tokens(const std::vector<std::string>& t, size_t& i,
                           const std::map<std::string, double>& env) {
    if (i >= t.size()) return 0.0;
    if (t[i] == "(") {
        i++;
        std::string op = t[i++];
        std::vector<double> args;
        while (i < t.size() && t[i] != ")") args.push_back(_eval_tokens(t, i, env));
        if (i < t.size()) i++;  // skip ')'
        double a = args.size() > 0 ? args[0] : 0.0;
        double b = args.size() > 1 ? args[1] : 0.0;
        if (op == "+") return a + b;
        if (op == "-") return a - b;
        if (op == "*") return a * b;
        if (op == "/") return b != 0.0 ? a / b : 0.0;
        return 0.0;
    }
    std::string a = t[i++];
    try { size_t p; double v = std::stod(a, &p); if (p == a.size()) return v; } catch (...) {}
    auto it = env.find(a);
    return it != env.end() ? it->second : 0.0;
}

// Evaluate an S-expression like "(* (+ a b) c)" under env.
inline double eval_sexpr(const std::string& s, const std::map<std::string, double>& env) {
    auto t = _sexpr_tokens(s);
    size_t i = 0;
    return _eval_tokens(t, i, env);
}

}  // namespace brain2
