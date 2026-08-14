#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <variant>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace brain3 {
namespace engines {
namespace synthesis {

// ── Symbolic expression (recursive variant) ──────────────────────────────────
// We represent expressions as a tagged-union tree:
//   Literal(double), Var("x"), UnaryOp(op, child), BinaryOp(op, left, right)
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Literal  { double val; };
struct Var      { std::string name; };
struct UnaryOp  { std::string op; ExprPtr child; };
struct BinaryOp { std::string op; ExprPtr left, right; };

struct Expr : std::variant<Literal, Var, UnaryOp, BinaryOp> {
    using variant::variant;
};

// Helpers
inline ExprPtr lit(double v)                        { return std::make_shared<Expr>(Literal{v}); }
inline ExprPtr var(const std::string& n = "x")     { return std::make_shared<Expr>(Var{n}); }
inline ExprPtr unary(const std::string& op, ExprPtr c)    { return std::make_shared<Expr>(UnaryOp{op, c}); }
inline ExprPtr binop(const std::string& op, ExprPtr l, ExprPtr r) { return std::make_shared<Expr>(BinaryOp{op, l, r}); }

// ── Evaluator ─────────────────────────────────────────────────────────────────
double eval_expr(const ExprPtr& e, double x) {
    return std::visit([&](auto&& v) -> double {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Literal>)  return v.val;
        if constexpr (std::is_same_v<T, Var>)      return (v.name == "x") ? x : throw std::runtime_error("unknown var");
        if constexpr (std::is_same_v<T, UnaryOp>) {
            double u = eval_expr(v.child, x);
            if (v.op == "neg") return -u;
            if (v.op == "sin") return std::sin(u);
            if (v.op == "cos") return std::cos(u);
            if (v.op == "exp") return std::exp(u);
            if (v.op == "ln")  return std::log(u);
            throw std::runtime_error("unknown unary op: " + v.op);
        }
        if constexpr (std::is_same_v<T, BinaryOp>) {
            double a = eval_expr(v.left, x), b = eval_expr(v.right, x);
            if (v.op == "+") return a + b;
            if (v.op == "-") return a - b;
            if (v.op == "*") return a * b;
            if (v.op == "/") return a / b;
            if (v.op == "^") return std::pow(a, b);
            throw std::runtime_error("unknown binary op: " + v.op);
        }
        throw std::runtime_error("unknown expr type");
    }, *e);
}

// ── Numerical derivative (central difference — the oracle) ───────────────────
double numerical_diff(const ExprPtr& e, double x, double h = 1e-7) {
    return (eval_expr(e, x + h) - eval_expr(e, x - h)) / (2 * h);
}

// ── Verification: candidate matches numerical derivative at multiple points ───
bool verify_rule(const ExprPtr& original, const ExprPtr& candidate,
                 double tol = 1e-4,
                 const std::vector<double>& pts = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.3, 1.7})
{
    for (double x : pts) {
        try {
            double num_d = numerical_diff(original, x);
            double sym_d = eval_expr(candidate, x);
            if (std::abs(num_d - sym_d) > tol * (std::abs(num_d) + 1.0)) return false;
        } catch (...) { return false; }
    }
    return true;
}

// ── CalculusSynth: discover differentiation rules by search + verification ────
class CalculusSynth {
public:
    struct LearnedRule {
        std::string name;
        ExprPtr fixed_expr;                              // for constant-form rules
        std::function<ExprPtr(int)> power_fn;            // for power rule: f(n) -> expr
        std::function<ExprPtr(ExprPtr, ExprPtr)> sum_fn; // for sum/product rules
    };

    std::map<std::string, LearnedRule> learned_rules;
    std::map<std::string, int> search_log;

    void learn(bool verbose = false) {
        _learn_power(verbose);
        _learn_trig("sin", verbose);
        _learn_trig("cos", verbose);
        _learn_simple_exp(verbose);
        _learn_simple_ln(verbose);
        _learn_sum_rule(verbose);
        _learn_product_rule(verbose);

        if (verbose) {
            int ok = (int)learned_rules.size();
            std::cout << "\nDiscovered " << ok << " rules from numerical verification.\n";
        }
    }

private:
    void _learn_power(bool verbose) {
        // Candidates for d/dx(x^n): try pattern over n=2,3,4,5
        struct PowerCand { std::string name; std::function<ExprPtr(int)> fn; };
        std::vector<PowerCand> cands = {
            {"n*x^(n-1)", [](int n){ return binop("*", lit(n), binop("^", var(), lit(n-1))); }},
            {"x^n",       [](int n){ return binop("^", var(), lit(n)); }},
            {"n*x^n",     [](int n){ return binop("*", lit(n), binop("^", var(), lit(n))); }},
            {"x^(n-1)",   [](int n){ return binop("^", var(), lit(n-1)); }},
            {"n*x",       [](int n){ return binop("*", lit(n), var()); }},
            {"1",         [](int){ return lit(1); }},
            {"0",         [](int){ return lit(0); }},
        };
        std::vector<int> test_ns = {2, 3, 4, 5};
        int tried = 0;
        for (const auto& c : cands) {
            bool all_ok = true;
            for (int n : test_ns) {
                tried++;
                auto original = binop("^", var(), lit(n));
                auto candidate = c.fn(n);
                if (!verify_rule(original, candidate)) { all_ok = false; break; }
            }
            if (all_ok) {
                learned_rules["power"] = {c.name, nullptr, c.fn};
                search_log["power"] = tried;
                if (verbose) std::cout << "  ✓ power rule: d/dx(x^n) = " << c.name << "\n";
                return;
            }
        }
        if (verbose) std::cout << "  ✗ power rule: FAILED\n";
    }

    void _learn_trig(const std::string& op, bool verbose) {
        struct TrigCand { std::string name; ExprPtr expr; };
        std::vector<TrigCand> cands;
        if (op == "sin") {
            cands = {
                {"cos(x)",   unary("cos", var())},
                {"-sin(x)",  unary("neg", unary("sin", var()))},
                {"sin(x)",   unary("sin", var())},
                {"-cos(x)",  unary("neg", unary("cos", var()))},
                {"1", lit(1)}, {"0", lit(0)},
            };
        } else {
            cands = {
                {"-sin(x)",  unary("neg", unary("sin", var()))},
                {"cos(x)",   unary("cos", var())},
                {"sin(x)",   unary("sin", var())},
                {"-cos(x)",  unary("neg", unary("cos", var()))},
                {"1", lit(1)}, {"0", lit(0)},
            };
        }
        auto original = unary(op, var());
        for (int i = 0; i < (int)cands.size(); i++) {
            if (verify_rule(original, cands[i].expr)) {
                learned_rules[op] = {cands[i].name, cands[i].expr};
                search_log[op] = i + 1;
                if (verbose) std::cout << "  ✓ " << op << " rule: d/dx(" << op << "(x)) = " << cands[i].name << "\n";
                return;
            }
        }
        if (verbose) std::cout << "  ✗ " << op << " rule: FAILED\n";
    }

    void _learn_simple_exp(bool verbose) {
        struct C { std::string name; ExprPtr e; };
        std::vector<C> cands = {
            {"exp(x)", unary("exp", var())},
            {"x*exp(x)", binop("*", var(), unary("exp", var()))},
            {"1", lit(1)}, {"0", lit(0)},
        };
        auto original = unary("exp", var());
        for (int i = 0; i < (int)cands.size(); i++) {
            if (verify_rule(original, cands[i].e)) {
                learned_rules["exp"] = {cands[i].name, cands[i].e};
                search_log["exp"] = i + 1;
                if (verbose) std::cout << "  ✓ exp rule: d/dx(exp(x)) = " << cands[i].name << "\n";
                return;
            }
        }
    }

    void _learn_simple_ln(bool verbose) {
        struct C { std::string name; ExprPtr e; };
        std::vector<C> cands = {
            {"1/x", binop("/", lit(1), var())},
            {"x",   var()}, {"ln(x)", unary("ln", var())},
            {"1", lit(1)}, {"0", lit(0)},
        };
        auto original = unary("ln", var());
        std::vector<double> pts = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0};
        for (int i = 0; i < (int)cands.size(); i++) {
            if (verify_rule(original, cands[i].e, 1e-4, pts)) {
                learned_rules["ln"] = {cands[i].name, cands[i].e};
                search_log["ln"] = i + 1;
                if (verbose) std::cout << "  ✓ ln rule: d/dx(ln(x)) = " << cands[i].name << "\n";
                return;
            }
        }
    }

    void _learn_sum_rule(bool verbose) {
        // d/dx(u + v) = du + dv  (discovered structurally)
        // Verified by: u = x^2, v = x^3, compare numerical diff to du + dv
        struct SumCand {
            std::string name;
            std::function<ExprPtr(ExprPtr, ExprPtr)> fn;
        };
        std::vector<SumCand> cands = {
            {"du + dv", [](ExprPtr du, ExprPtr dv){ return binop("+", du, dv); }},
            {"du * dv", [](ExprPtr du, ExprPtr dv){ return binop("*", du, dv); }},
            {"du",      [](ExprPtr du, ExprPtr){ return du; }},
            {"dv",      [](ExprPtr, ExprPtr dv){ return dv; }},
        };
        // u = x^2 (du = 2x), v = x^3 (dv = 3x^2), sum = x^2 + x^3
        auto u = binop("^", var(), lit(2));
        auto v = binop("^", var(), lit(3));
        auto original = binop("+", u, v);
        auto du = binop("*", lit(2), var());
        auto dv = binop("*", lit(3), binop("^", var(), lit(2)));
        for (int i = 0; i < (int)cands.size(); i++) {
            auto cand_expr = cands[i].fn(du, dv);
            if (verify_rule(original, cand_expr)) {
                learned_rules["sum"] = {cands[i].name, nullptr, nullptr, cands[i].fn};
                search_log["sum"] = i + 1;
                if (verbose) std::cout << "  ✓ sum rule: d/dx(u+v) = " << cands[i].name << "\n";
                return;
            }
        }
    }

    void _learn_product_rule(bool verbose) {
        // d/dx(u * v) = du*v + u*dv  (product rule)
        struct ProdCand {
            std::string name;
            std::function<ExprPtr(ExprPtr, ExprPtr, ExprPtr, ExprPtr)> fn;
        };
        std::vector<ProdCand> cands = {
            {"du*v + u*dv", [](ExprPtr u, ExprPtr v, ExprPtr du, ExprPtr dv){
                return binop("+", binop("*", du, v), binop("*", u, dv)); }},
            {"du * dv",     [](ExprPtr, ExprPtr, ExprPtr du, ExprPtr dv){ return binop("*", du, dv); }},
            {"u * dv",      [](ExprPtr u, ExprPtr, ExprPtr, ExprPtr dv){ return binop("*", u, dv); }},
            {"du * v",      [](ExprPtr, ExprPtr v, ExprPtr du, ExprPtr){ return binop("*", du, v); }},
        };
        // u=x^2, v=x^3, product=x^5, du=2x, dv=3x^2
        auto u = binop("^", var(), lit(2));
        auto v = binop("^", var(), lit(3));
        auto original = binop("*", u, v);  // x^5
        auto du = binop("*", lit(2), var());
        auto dv = binop("*", lit(3), binop("^", var(), lit(2)));
        for (int i = 0; i < (int)cands.size(); i++) {
            auto cand_expr = cands[i].fn(u, v, du, dv);
            if (verify_rule(original, cand_expr)) {
                learned_rules["product"] = {cands[i].name, nullptr, nullptr,
                    [fn=cands[i].fn](ExprPtr a, ExprPtr b){ return fn(a, b, a, b); }};
                search_log["product"] = i + 1;
                if (verbose) std::cout << "  ✓ product rule: d/dx(u*v) = " << cands[i].name << "\n";
                return;
            }
        }
    }

public:
    // Apply learned rules to differentiate an expression
    ExprPtr diff(const ExprPtr& e) {
        return std::visit([&](auto&& v) -> ExprPtr {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Literal>)  return lit(0.0);
            if constexpr (std::is_same_v<T, Var>)      return lit(1.0);
            if constexpr (std::is_same_v<T, UnaryOp>) {
                auto it = learned_rules.find(v.op);
                if (it != learned_rules.end() && it->second.fixed_expr)
                    return it->second.fixed_expr;
                return lit(0.0);  // rule not yet learned
            }
            if constexpr (std::is_same_v<T, BinaryOp>) {
                if (v.op == "+" || v.op == "-") {
                    auto it = learned_rules.find("sum");
                    if (it != learned_rules.end() && it->second.sum_fn) {
                        auto du = diff(v.left), dv = diff(v.right);
                        auto combined = it->second.sum_fn(du, dv);
                        if (v.op == "-") return binop("-", du, dv);
                        return combined;
                    }
                }
                if (v.op == "*") {
                    auto it = learned_rules.find("product");
                    if (it != learned_rules.end() && it->second.sum_fn) {
                        auto du = diff(v.left), dv = diff(v.right);
                        return it->second.sum_fn(v.left, v.right);
                    }
                    auto du = diff(v.left), dv = diff(v.right);
                    return binop("+", binop("*", du, v.right), binop("*", v.left, dv));
                }
                if (v.op == "^") {
                    // Extract n if right side is a literal
                    if (auto* lit_n = std::get_if<Literal>(v.right.get())) {
                        int n = (int)lit_n->val;
                        auto it = learned_rules.find("power");
                        if (it != learned_rules.end() && it->second.power_fn)
                            return it->second.power_fn(n);
                    }
                }
                return lit(0.0);
            }
            return lit(0.0);
        }, *e);
    }
};

}}}
