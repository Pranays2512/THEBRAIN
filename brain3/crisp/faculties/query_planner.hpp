#pragma once
#include <string>
#include <vector>

namespace brain2 {
namespace faculties {

// Rule-based translation of English into BrainQL
class QueryPlanner {
public:
    std::string plan_query(const std::string& text) {
        // Simple heuristic translation to BrainQL
        if (text.find("what is") != std::string::npos) {
            std::string entity = text.substr(text.find("what is") + 8);
            if (!entity.empty() && entity.back() == '?') entity.pop_back();
            return "MATCH " + entity + " * ?";
        }
        return "UNKNOWN QUERY";
    }
};

} // namespace faculties
} // namespace brain2
