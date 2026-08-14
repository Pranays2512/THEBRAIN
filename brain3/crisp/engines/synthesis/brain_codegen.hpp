#pragma once
#include <string>
#include <vector>
#include <memory>
#include "../../../fuzzy/engines/synthesis/policy_induction.hpp"

namespace brain3 { 
namespace engines { 
namespace synthesis {

// The brain builds the LOGIC; rendering is mechanical (no LLM).
class BrainCodeGen {
public:
    std::string render_cpp_expr(std::shared_ptr<brain2::synthesis::ExprNode> e) {
        if (!e) return "";
        if (e->op == "VAR" || e->op == "CONST") {
            return e->val;
        }
        
        std::string l = render_cpp_expr(e->left);
        std::string r = render_cpp_expr(e->right);
        
        if (e->op == "neg") return "(-" + l + ")";
        
        // Render powers (since C++ does not have **)
        if (e->op == "^" || e->op == "**") {
            return "std::pow(" + l + ", " + r + ")";
        }
        
        return "(" + l + " " + e->op + " " + r + ")";
    }

    std::string to_cpp_function(const std::string& fn_name, const std::vector<std::string>& inputs, std::shared_ptr<brain2::synthesis::ExprNode> expr) {
        std::string code = "double " + fn_name + "(";
        for (size_t i = 0; i < inputs.size(); i++) {
            code += "double " + inputs[i];
            if (i < inputs.size() - 1) code += ", ";
        }
        code += ") {\n    return " + render_cpp_expr(expr) + ";\n}\n";
        return code;
    }
};

}}}
