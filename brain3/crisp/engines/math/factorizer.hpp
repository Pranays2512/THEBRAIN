#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include "calculus_engine.hpp" // For ExprNode

namespace brain2 {
namespace math {

// A simplified implementation of anti-unification (shape matching) over ASTs
inline std::shared_ptr<ExprNode> antiunify(std::shared_ptr<ExprNode> a, std::shared_ptr<ExprNode> b, int& hole_ctr) {
    if (!a->is_leaf() && !b->is_leaf() && a->op == b->op && a->children.size() == b->children.size()) {
        std::vector<std::shared_ptr<ExprNode>> c;
        for (size_t i = 0; i < a->children.size(); ++i) {
            c.push_back(antiunify(a->children[i], b->children[i], hole_ctr));
        }
        return ExprNode::make_op(a->op, c);
    }
    
    if (a->op == b->op && a->val == b->val && a->var == b->var) {
        if (a->op == "val") return ExprNode::make_num(a->val);
        return ExprNode::make_var(a->var);
    }
    
    return ExprNode::make_var("hole_" + std::to_string(hole_ctr++));
}

// Discover the shared skeleton between two formulas
inline std::shared_ptr<ExprNode> factor_au(std::shared_ptr<ExprNode> a, std::shared_ptr<ExprNode> b) {
    int ctr = 0;
    return antiunify(a, b, ctr);
}

} // namespace math
} // namespace brain2
