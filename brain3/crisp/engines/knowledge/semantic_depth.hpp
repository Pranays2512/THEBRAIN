#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <regex>

namespace brain2 {
namespace knowledge {

// Mock structure for demonstration of semantic depth verification
struct ParsedDef {
    std::string target;
    std::string op;
    std::string arg1;
    std::string arg2;
};

inline std::optional<ParsedDef> learn_definition(const std::string& text, const std::set<std::string>& known_relations) {
    std::map<std::string, std::string> DEF_OP = {
        {"times", "*"}, {"multiplied", "*"}, {"product", "*"},
        {"divided", "/"}, {"over", "/"}, {"per", "/"},
        {"plus", "+"}, {"minus", "-"}
    };
    
    std::regex word_re(R"([a-z_]+)");
    std::string s = text;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    
    std::sregex_token_iterator it(s.begin(), s.end(), word_re);
    std::sregex_token_iterator end;
    std::vector<std::string> toks;
    for (; it != end; ++it) toks.push_back(*it);
    
    auto is_it = std::find(toks.begin(), toks.end(), "is");
    if (is_it == toks.end()) return std::nullopt;
    
    int is_idx = std::distance(toks.begin(), is_it);
    if (is_idx == 0) return std::nullopt;
    
    std::string target = toks[is_idx - 1];
    std::string op = "";
    std::vector<std::string> args;
    
    for (int i = is_idx + 1; i < (int)toks.size(); ++i) {
        if (DEF_OP.count(toks[i])) op = DEF_OP[toks[i]];
        if (known_relations.count(toks[i])) args.push_back(toks[i]);
    }
    
    if (op.empty() || args.size() < 2) return std::nullopt;
    
    // In a real integration, here we would add to the PolicyPack and test with Solver.
    return ParsedDef{target, op, args[0], args[1]};
}

} // namespace knowledge
} // namespace brain2
