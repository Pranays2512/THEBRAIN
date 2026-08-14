#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>
#include <iostream>

namespace brain2 {
namespace math {

// Very simple recursive AST node for symbolic expressions
struct ExprNode {
    std::string op; // Operator or "val" for leaves
    double val = 0.0;
    std::string var = "";
    std::vector<std::shared_ptr<ExprNode>> children;
    
    bool is_leaf() const { return op == "val" || op == "var"; }
    
    static std::shared_ptr<ExprNode> make_num(double v) {
        auto n = std::make_shared<ExprNode>();
        n->op = "val"; n->val = v; return n;
    }
    static std::shared_ptr<ExprNode> make_var(const std::string& v) {
        auto n = std::make_shared<ExprNode>();
        n->op = "var"; n->var = v; return n;
    }
    static std::shared_ptr<ExprNode> make_op(const std::string& o, const std::vector<std::shared_ptr<ExprNode>>& c) {
        auto n = std::make_shared<ExprNode>();
        n->op = o; n->children = c; return n;
    }
};

inline bool is_num(const std::shared_ptr<ExprNode>& e) { return e && e->op == "val"; }
inline bool is_var(const std::shared_ptr<ExprNode>& e) { return e && e->op == "var"; }

inline std::shared_ptr<ExprNode> simplify(std::shared_ptr<ExprNode> e) {
    if (!e || e->is_leaf()) return e;
    std::vector<std::shared_ptr<ExprNode>> args;
    for (auto c : e->children) args.push_back(simplify(c));
    
    if (e->op == "neg") {
        if (is_num(args[0])) return ExprNode::make_num(-args[0]->val);
        if (args[0]->op == "neg") return args[0]->children[0];
        return ExprNode::make_op("neg", args);
    }
    
    if (args.size() < 2) return ExprNode::make_op(e->op, args);
    auto a = args[0];
    auto b = args[1];
    
    if (e->op == "+") {
        if (is_num(a) && a->val == 0) return b;
        if (is_num(b) && b->val == 0) return a;
        if (is_num(a) && is_num(b)) return ExprNode::make_num(a->val + b->val);
    } else if (e->op == "-") {
        if (is_num(b) && b->val == 0) return a;
        if (is_num(a) && is_num(b)) return ExprNode::make_num(a->val - b->val);
    } else if (e->op == "*") {
        if ((is_num(a) && a->val == 0) || (is_num(b) && b->val == 0)) return ExprNode::make_num(0);
        if (is_num(a) && a->val == 1) return b;
        if (is_num(b) && b->val == 1) return a;
        if (is_num(a) && is_num(b)) return ExprNode::make_num(a->val * b->val);
        // c * (u / d) -> (c/d) * u
        if (is_num(a) && b->op == "/" && is_num(b->children[1])) {
            double coeff = a->val / b->children[1]->val;
            if (coeff == 1.0) return b->children[0];
            return simplify(ExprNode::make_op("*", {ExprNode::make_num(coeff), b->children[0]}));
        }
        // c * (d * u) -> (c*d) * u
        if (is_num(a) && b->op == "*" && is_num(b->children[0])) {
            return simplify(ExprNode::make_op("*", {ExprNode::make_num(a->val * b->children[0]->val), b->children[1]}));
        }
        // (d * u) * c -> (c*d) * u
        if (is_num(b) && a->op == "*" && is_num(a->children[0])) {
            return simplify(ExprNode::make_op("*", {ExprNode::make_num(b->val * a->children[0]->val), a->children[1]}));
        }
    } else if (e->op == "/") {
        if (is_num(a) && a->val == 0) return ExprNode::make_num(0);
        if (is_num(b) && b->val == 1) return a;
        if (is_num(a) && is_num(b) && b->val != 0) return ExprNode::make_num(a->val / b->val);
        // (c * u) / d -> (c/d) * u
        if (is_num(b) && a->op == "*" && is_num(a->children[0]) && b->val != 0) {
            double coeff = a->children[0]->val / b->val;
            if (coeff == 1.0) return a->children[1];
            return simplify(ExprNode::make_op("*", {ExprNode::make_num(coeff), a->children[1]}));
        }
    } else if (e->op == "^") {
        if (is_num(b) && b->val == 1) return a;
        if (is_num(b) && b->val == 0) return ExprNode::make_num(1);
    }
    return ExprNode::make_op(e->op, args);
}

inline std::string render(std::shared_ptr<ExprNode> e) {
    if (!e) return "";
    if (e->op == "val") {
        std::string s = std::to_string(e->val);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (s.back() == '.') s.pop_back();
        return s;
    }
    if (e->op == "var") return e->var;
    if (e->op == "neg") return "-" + render(e->children[0]);
    if (e->op == "sin" || e->op == "cos" || e->op == "exp" || e->op == "ln") {
        return e->op + "(" + render(e->children[0]) + ")";
    }
    std::string a = render(e->children[0]);
    std::string b = render(e->children[1]);
    
    auto paren = [](std::shared_ptr<ExprNode> n, const std::string& s) {
        if (n->is_leaf() || n->op == "sin" || n->op == "cos" || n->op == "exp" || n->op == "ln" || n->op == "^" || n->op == "neg") {
            return s;
        }
        return "(" + s + ")";
    };
    
    if (e->op == "^") return paren(e->children[0], a) + "^" + b;
    std::string sym = e->op == "+" ? " + " : e->op == "-" ? " - " : e->op;
    return paren(e->children[0], a) + sym + paren(e->children[1], b);
}

using ExprPtr = std::shared_ptr<ExprNode>;

class CalculusEngine {
public:
    /**
     * Compute analytical symbolic derivative d/dvar (e).
     */
    static ExprPtr diff(const ExprPtr& e, const std::string& var = "x") {
        if (!e) return ExprNode::make_num(0);
        if (is_num(e)) return ExprNode::make_num(0);
        if (is_var(e)) return (e->var == var) ? ExprNode::make_num(1) : ExprNode::make_num(0);

        const auto& op = e->op;
        const auto& ch = e->children;

        if (op == "neg") return simplify(ExprNode::make_op("neg", {diff(ch[0], var)}));
        if (op == "+")   return simplify(ExprNode::make_op("+", {diff(ch[0], var), diff(ch[1], var)}));
        if (op == "-")   return simplify(ExprNode::make_op("-", {diff(ch[0], var), diff(ch[1], var)}));
        
        // Product Rule: (f * g)' = f' * g + f * g'
        if (op == "*") {
            auto term1 = ExprNode::make_op("*", {diff(ch[0], var), ch[1]});
            auto term2 = ExprNode::make_op("*", {ch[0], diff(ch[1], var)});
            return simplify(ExprNode::make_op("+", {term1, term2}));
        }

        // Quotient Rule: (f / g)' = (f' * g - f * g') / (g ^ 2)
        if (op == "/") {
            auto num_term1 = ExprNode::make_op("*", {diff(ch[0], var), ch[1]});
            auto num_term2 = ExprNode::make_op("*", {ch[0], diff(ch[1], var)});
            auto numerator = ExprNode::make_op("-", {num_term1, num_term2});
            auto denominator = ExprNode::make_op("^", {ch[1], ExprNode::make_num(2)});
            return simplify(ExprNode::make_op("/", {numerator, denominator}));
        }

        // Power Rule with General Chain Rule: (u(x) ^ n)' = n * u(x)^(n-1) * u'(x)
        if (op == "^") {
            if (is_num(ch[1])) {
                double n = ch[1]->val;
                auto pow_term = ExprNode::make_op("^", {ch[0], ExprNode::make_num(n - 1)});
                auto scaled_pow = ExprNode::make_op("*", {ExprNode::make_num(n), pow_term});
                return simplify(ExprNode::make_op("*", {scaled_pow, diff(ch[0], var)}));
            }
        }

        // Chain Rule: (sin(u))' = cos(u) * u'
        if (op == "sin") {
            auto cos_term = ExprNode::make_op("cos", {ch[0]});
            return simplify(ExprNode::make_op("*", {cos_term, diff(ch[0], var)}));
        }

        // Chain Rule: (cos(u))' = -sin(u) * u'
        if (op == "cos") {
            auto neg_sin = ExprNode::make_op("neg", {ExprNode::make_op("sin", {ch[0]})});
            return simplify(ExprNode::make_op("*", {neg_sin, diff(ch[0], var)}));
        }

        // Chain Rule: (exp(u))' = exp(u) * u'
        if (op == "exp") {
            auto exp_term = ExprNode::make_op("exp", {ch[0]});
            return simplify(ExprNode::make_op("*", {exp_term, diff(ch[0], var)}));
        }

        // Chain Rule: (ln(u))' = u' / u
        if (op == "ln") {
            return simplify(ExprNode::make_op("/", {diff(ch[0], var), ch[0]}));
        }

        return ExprNode::make_num(0);
    }

    /**
     * Compute n-th order symbolic derivative d^n / dvar^n (e).
     */
    static ExprPtr diff_n(const ExprPtr& e, const std::string& var, int n) {
        ExprPtr cur = e;
        for (int i = 0; i < n; ++i) {
            cur = diff(cur, var);
        }
        return cur;
    }

    /**
     * Numerical evaluation of expression under variable environment.
     */
    static double eval(const ExprPtr& e, const std::map<std::string, double>& env) {
        if (!e) return 0.0;
        if (is_num(e)) return e->val;
        if (is_var(e)) {
            auto it = env.find(e->var);
            if (it != env.end()) return it->second;
            throw std::runtime_error("Undefined variable: " + e->var);
        }
        const auto& op = e->op;
        if (op == "neg") return -eval(e->children[0], env);
        if (op == "sin") return std::sin(eval(e->children[0], env));
        if (op == "cos") return std::cos(eval(e->children[0], env));
        if (op == "exp") return std::exp(eval(e->children[0], env));
        if (op == "ln")  return std::log(eval(e->children[0], env));
        
        double a = eval(e->children[0], env);
        double b = eval(e->children[1], env);
        if (op == "+") return a + b;
        if (op == "-") return a - b;
        if (op == "*") return a * b;
        if (op == "/") {
            if (std::abs(b) < 1e-12) throw std::runtime_error("Division by zero in eval");
            return a / b;
        }
        if (op == "^") return std::pow(a, b);
        return 0.0;
    }

    /**
     * Numerically verify symbolic derivative against finite-difference quotient.
     */
    static bool verify_derivative(const ExprPtr& f, const ExprPtr& df,
                                  const std::string& var = "x",
                                  double x0 = 1.5, double h = 1e-6, double tol = 1e-4) {
        try {
            double sym_val = eval(df, {{var, x0}});
            double f_plus = eval(f, {{var, x0 + h}});
            double f_minus = eval(f, {{var, x0 - h}});
            double num_val = (f_plus - f_minus) / (2.0 * h);
            return std::abs(sym_val - num_val) < tol;
        } catch (...) {
            return false;
        }
    }
};

} // namespace math
} // namespace brain2
