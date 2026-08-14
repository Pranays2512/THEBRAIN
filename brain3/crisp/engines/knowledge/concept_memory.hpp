#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>

namespace brain2 {
namespace knowledge {

// A simple recursive string-based AST for shape matching
struct ShapeNode {
    std::string val;
    std::vector<std::shared_ptr<ShapeNode>> children;
    
    static std::shared_ptr<ShapeNode> make(const std::string& v, std::vector<std::shared_ptr<ShapeNode>> c = {}) {
        auto n = std::make_shared<ShapeNode>();
        n->val = v;
        n->children = c;
        return n;
    }
    
    bool is_var() const {
        if (val.empty()) return false;
        for (char c : val) {
            if (std::islower(c)) return false; // Contains lowercase, not purely UPPERCASE variable
        }
        return true; // Is uppercase variable
    }
};

inline bool match_shape(const std::shared_ptr<ShapeNode>& shape, const std::shared_ptr<ShapeNode>& expr, std::map<std::string, std::shared_ptr<ShapeNode>>& bind) {
    if (shape->is_var()) {
        if (bind.count(shape->val)) {
            // Very simplified equality check
            if (bind[shape->val]->val != expr->val) return false;
        } else {
            bind[shape->val] = expr;
        }
        return true;
    }
    
    if (shape->children.size() == expr->children.size() && shape->val == expr->val) {
        for (size_t i = 0; i < shape->children.size(); ++i) {
            if (!match_shape(shape->children[i], expr->children[i], bind)) return false;
        }
        return true;
    }
    return false;
}

struct ConceptInfo {
    std::shared_ptr<ShapeNode> shape;
    int uses = 0;
    std::string status = "candidate";
};

class ConceptMemory {
public:
    std::map<std::string, ConceptInfo> concepts;
    int promote_at;
    int n_count = 0;
    
    ConceptMemory(int pa = 3) : promote_at(pa) {}
    
    std::string register_concept(const std::shared_ptr<ShapeNode>& shape) {
        // dedupe
        for (const auto& kv : concepts) {
            if (kv.second.shape->val == shape->val && kv.second.shape->children.size() == shape->children.size()) {
                // Not doing deep equality here for brevity, but this is a proxy
                return kv.first;
            }
        }
        std::string name = "concept_" + std::to_string(n_count++);
        concepts[name] = {shape, 0, "candidate"};
        return name;
    }
    
    std::pair<std::string, std::map<std::string, std::shared_ptr<ShapeNode>>> recognize(const std::shared_ptr<ShapeNode>& expr) {
        for (const auto& kv : concepts) {
            std::map<std::string, std::shared_ptr<ShapeNode>> bind;
            if (match_shape(kv.second.shape, expr, bind)) {
                return {kv.first, bind};
            }
        }
        return {"", {}};
    }
    
    void record_use(const std::string& name) {
        if (concepts.count(name) == 0) return;
        concepts[name].uses++;
        if (concepts[name].uses >= promote_at && concepts[name].status == "candidate") {
            concepts[name].status = "promoted";
        }
    }
};

} // namespace knowledge
} // namespace brain2
