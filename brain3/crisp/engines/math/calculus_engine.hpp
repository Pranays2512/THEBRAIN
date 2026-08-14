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
    } else if (e->op == "/") {
        if (is_num(a) && a->val == 0) return ExprNode::make_num(0);
        if (is_num(b) && b->val == 1) return a;
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

} // namespace math
} // namespace brain2
