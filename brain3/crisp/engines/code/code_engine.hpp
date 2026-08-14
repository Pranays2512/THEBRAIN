#pragma once

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <map>
#include <set>
#include <functional>
#include <iostream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <sstream>

namespace brain2 {

// Dynamic type representation for DSL state
struct State {
    enum Type { EMPTY, INT, BOOL, INT_LIST, PAIR_LIST, NESTED_INT_LIST };
    Type type;
    
    int int_val = 0;
    bool bool_val = false;
    std::vector<int> int_list;
    std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> pair_list;
    std::vector<std::vector<int>> nested_int_list;

    State() : type(EMPTY) {}
    State(int v) : type(INT), int_val(v) {}
    State(bool v) : type(BOOL), bool_val(v) {}
    State(const std::vector<int>& v) : type(INT_LIST), int_list(v) {}
    State(const std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>>& v) : type(PAIR_LIST), pair_list(v) {}
    State(const std::vector<std::vector<int>>& v) : type(NESTED_INT_LIST), nested_int_list(v) {}

    bool operator==(const State& o) const {
        if (type != o.type) return false;
        switch (type) {
            case EMPTY: return true;
            case INT: return int_val == o.int_val;
            case BOOL: return bool_val == o.bool_val;
            case INT_LIST: return int_list == o.int_list;
            case PAIR_LIST: return pair_list == o.pair_list;
            case NESTED_INT_LIST: return nested_int_list == o.nested_int_list;
        }
        return false;
    }
    bool operator!=(const State& o) const { return !(*this == o); }

    std::string signature() const {
        if (type == EMPTY) return "E";
        if (type == INT) return "I" + std::to_string(int_val);
        if (type == BOOL) return "B" + std::to_string(bool_val);
        if (type == INT_LIST) {
            std::string s = "L[";
            for (int x : int_list) s += std::to_string(x) + ",";
            return s + "]";
        }
        if (type == PAIR_LIST) {
            std::string s = "P[";
            for (auto& p : pair_list) {
                s += "(" + std::to_string(p.first.first) + "," + std::to_string(p.first.second) + ")-" +
                     "(" + std::to_string(p.second.first) + "," + std::to_string(p.second.second) + "),";
            }
            return s + "]";
        }
        if (type == NESTED_INT_LIST) {
            std::string s = "N[";
            for (auto& l : nested_int_list) {
                s += "[";
                for (int x : l) s += std::to_string(x) + ",";
                s += "],";
            }
            return s + "]";
        }
        return "?";
    }
};

struct ASTNode {
    std::string op_type;
    std::string op_name;
    std::shared_ptr<ASTNode> child;

    ASTNode(std::string t, std::string n, std::shared_ptr<ASTNode> c)
        : op_type(t), op_name(n), child(c) {}
};

struct SynthesisResult {
    std::shared_ptr<ASTNode> tree;
    std::string code;
    std::string code_java;
    std::string code_cpp;
    int candidates_searched = 0;
};

class CodeEngine {
public:
    int target_val;

    CodeEngine(int target = 0) : target_val(target) {}

    // A* Heuristic
    float heuristic(const std::vector<State>& current, const std::vector<State>& goal) {
        float total_dist = 0;
        for (size_t i = 0; i < current.size(); i++) {
            const auto& curr = current[i];
            const auto& g = goal[i];

            if (g.type == State::BOOL) {
                if (curr.type == State::BOOL && curr.bool_val == g.bool_val) continue;
                else if (curr.type == State::BOOL) total_dist += 10;
                else return INFINITY;
            } else if (g.type == State::INT_LIST) {
                if (curr.type == State::NESTED_INT_LIST) {
                    bool found = false;
                    for (auto& l : curr.nested_int_list) {
                        if (l == g.int_list) { found = true; break; }
                    }
                    if (found) total_dist += 1.0f;
                    else total_dist += 20.0f;
                } else if (curr.type == State::PAIR_LIST) {
                    int len_diff = std::abs((int)curr.pair_list.size() - (int)g.int_list.size());
                    total_dist += len_diff * 1.0f + 5.0f; 
                } else if (curr.type != State::INT_LIST) {
                    return INFINITY;
                } else {
                    int len_diff = std::abs((int)curr.int_list.size() - (int)g.int_list.size());
                    total_dist += len_diff * 1.0f;

                    if (curr.int_list.size() == g.int_list.size() && curr.int_list.size() > 0) {
                        auto c_sort = curr.int_list; std::sort(c_sort.begin(), c_sort.end());
                        auto g_sort = g.int_list;    std::sort(g_sort.begin(), g_sort.end());
                        for (size_t k = 0; k < c_sort.size(); k++) {
                            total_dist += std::abs(c_sort[k] - g_sort[k]);
                        }
                    }
                    if (curr != g) total_dist += 5.0f;
                }
            } else {
                if (curr != g) return INFINITY;
            }
        }
        return total_dist;
    }

    State apply_op(const std::string& op_type, const std::string& op_name, const State& data) {
        if (data.type != State::INT_LIST && data.type != State::PAIR_LIST && data.type != State::NESTED_INT_LIST) return State();

        try {
            if (op_type == "Map") {
                if (data.type == State::INT_LIST) {
                    std::vector<int> res; res.reserve(data.int_list.size());
                    for (int x : data.int_list) {
                        if (op_name == "*2") res.push_back(x * 2);
                        else if (op_name == "+1") res.push_back(x + 1);
                        else if (op_name == "-1") res.push_back(x - 1);
                        else if (op_name == "^2") res.push_back(x * x);
                        else if (op_name == "abs") res.push_back(std::abs(x));
                        else return State();
                    }
                    return State(res);
                } else if (data.type == State::PAIR_LIST && op_name == "extract_indices") {
                    std::vector<std::vector<int>> res;
                    for (const auto& p : data.pair_list) {
                        res.push_back({p.first.first, p.second.first});
                    }
                    return State(res);
                }
            } else if (op_type == "Filter") {
                if (data.type == State::INT_LIST) {
                    std::vector<int> res;
                    for (int x : data.int_list) {
                        bool keep = false;
                        if (op_name == ">0") keep = (x > 0);
                        else if (op_name == "<0") keep = (x < 0);
                        else if (op_name == "even") keep = (x % 2 == 0);
                        else if (op_name == "odd") keep = (x % 2 != 0);
                        if (keep) res.push_back(x);
                    }
                    return State(res);
                } else if (data.type == State::PAIR_LIST && op_name == "sum==target & i!=j") {
                    std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> res;
                    for (const auto& p : data.pair_list) {
                        if (p.first.first != p.second.first && (p.first.second + p.second.second == target_val)) {
                            res.push_back(p);
                        }
                    }
                    return State(res);
                }
            } else if (op_type == "List" && data.type == State::INT_LIST) {
                if (op_name == "Reverse") {
                    auto res = data.int_list;
                    std::reverse(res.begin(), res.end());
                    return State(res);
                } else if (op_name == "Sort") {
                    auto res = data.int_list;
                    std::sort(res.begin(), res.end());
                    return State(res);
                } else if (op_name == "ToSet") {
                    std::vector<int> res;
                    std::set<int> s;
                    for (int x : data.int_list) {
                        if (s.insert(x).second) res.push_back(x);
                    }
                    return State(res);
                } else if (op_name == "CumulativeSum") {
                    auto res = data.int_list;
                    for (size_t i = 1; i < res.size(); i++) res[i] += res[i-1];
                    return State(res);
                } else if (op_name == "EnumProduct") {
                    std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> res;
                    for (size_t i = 0; i < data.int_list.size(); i++) {
                        for (size_t j = 0; j < data.int_list.size(); j++) {
                            res.push_back({{i, data.int_list[i]}, {j, data.int_list[j]}});
                        }
                    }
                    return State(res);
                } else if (op_name == "First") {
                    if (data.int_list.empty()) return State();
                    std::vector<int> res = {data.int_list[0]};
                    return State(res); // Simplified to list of 1 element for consistency
                }
            } else if (op_type == "List" && data.type == State::NESTED_INT_LIST) {
                if (op_name == "First") {
                    if (data.nested_int_list.empty()) return State();
                    return State(data.nested_int_list[0]);
                }
            } else if (op_type == "Terminal" && data.type == State::INT_LIST) {
                if (op_name == "HasDuplicate") {
                    std::set<int> s(data.int_list.begin(), data.int_list.end());
                    return State(s.size() != data.int_list.size());
                }
            }
        } catch (...) { }
        return State();
    }

    std::vector<std::string> map_ops_names = {"*2", "+1", "-1", "^2", "abs", "extract_indices"};
    std::vector<std::string> filter_ops_names = {">0", "<0", "even", "odd", "sum==target & i!=j"};
    std::vector<std::string> list_ops_names = {"Reverse", "Sort", "ToSet", "EnumProduct", "First", "CumulativeSum"};
    std::vector<std::string> terminal_ops_names = {"HasDuplicate"};

    // AST Renderers
    std::string render_python(std::shared_ptr<ASTNode> tree) {
        if (!tree) return "input_list";
        if (tree->op_type == "Input") return "input_list";

        if (tree->op_type == "Map") {
            std::string inner = render_python(tree->child);
            std::string expr = tree->op_name;
            if (expr == "*2") expr = "x * 2";
            else if (expr == "+1") expr = "x + 1";
            else if (expr == "-1") expr = "x - 1";
            else if (expr == "^2") expr = "x ** 2";
            else if (expr == "abs") expr = "abs(x)";
            else if (expr == "extract_indices") expr = "[x[0][0], x[1][0]]";
            return "[" + expr + " for x in " + inner + "]";
        } else if (tree->op_type == "Filter") {
            std::string inner = render_python(tree->child);
            std::string cond = tree->op_name;
            if (cond == ">0") cond = "x > 0";
            else if (cond == "<0") cond = "x < 0";
            else if (cond == "even") cond = "x % 2 == 0";
            else if (cond == "odd") cond = "x % 2 != 0";
            else if (cond == "sum==target & i!=j") cond = "x[0][0] != x[1][0] and x[0][1] + x[1][1] == target";
            return "[x for x in " + inner + " if " + cond + "]";
        } else {
            std::string inner = render_python(tree->child);
            if (tree->op_name == "Reverse") return "list(reversed(" + inner + "))";
            if (tree->op_name == "Sort") return "sorted(" + inner + ")";
            if (tree->op_name == "CumulativeSum") return "[sum(" + inner + "[:i+1]) for i in range(len(" + inner + "))]";
            if (tree->op_name == "EnumProduct") return "[((i, x), (j, y)) for i, x in enumerate(" + inner + ") for j, y in enumerate(" + inner + ")]";
            if (tree->op_name == "First") return "(" + inner + "[0] if " + inner + " and len(" + inner + ") >= 1 else None)";
            if (tree->op_name == "HasDuplicate") return "(len(set(" + inner + ")) != len(" + inner + "))";
        }
        return "unknown";
    }

    std::vector<std::pair<std::string, std::string>> flatten_tree(std::shared_ptr<ASTNode> tree) {
        std::vector<std::pair<std::string, std::string>> ops;
        auto curr = tree;
        while (curr && curr->op_type != "Input") {
            ops.push_back({curr->op_type, curr->op_name});
            curr = curr->child;
        }
        std::reverse(ops.begin(), ops.end());
        return ops;
    }

    std::string render_cpp(std::shared_ptr<ASTNode> tree) {
        if (!tree || tree->op_type == "Input") {
            return "std::vector<int> solve(std::vector<int> input) {\n    return input;\n}";
        }
        auto ops = flatten_tree(tree);
        std::string code = "std::vector<int> solve(std::vector<int> input) {\n";
        
        std::vector<std::pair<std::string, std::string>> fusable, non_fusable;
        size_t idx = 0;
        while (idx < ops.size() && (ops[idx].first == "Map" || ops[idx].first == "Filter")) {
            fusable.push_back(ops[idx]);
            idx++;
        }
        while (idx < ops.size()) {
            non_fusable.push_back(ops[idx]);
            idx++;
        }

        if (!fusable.empty()) {
            if (fusable.size() > 1) code += "    // ⚡ OPTIMIZED: Loop Fusion (Deforestation)\n";
            code += "    std::vector<int> res;\n    res.reserve(input.size());\n    for (auto& x : input) {\n";
            std::string indent = "        ";
            int open_braces = 0;
            for (auto& op : fusable) {
                if (op.first == "Filter") {
                    std::string cond = op.second;
                    if (cond == ">0") cond = "x > 0";
                    else if (cond == "<0") cond = "x < 0";
                    else if (cond == "even") cond = "x % 2 == 0";
                    else if (cond == "odd") cond = "x % 2 != 0";
                    else if (cond == "sum==target & i!=j") cond = "x.first.first != x.second.first && (x.first.second + x.second.second == target_val)";
                    code += indent + "if (" + cond + ") {\n";
                    indent += "    ";
                    open_braces++;
                } else if (op.first == "Map") {
                    std::string expr = op.second;
                    if (expr == "*2") expr = "x * 2";
                    else if (expr == "+1") expr = "x + 1";
                    else if (expr == "-1") expr = "x - 1";
                    else if (expr == "^2") expr = "x * x";
                    else if (expr == "abs") expr = "std::abs(x)";
                    else if (expr == "extract_indices") expr = "std::vector<int>{x.first.first, x.second.first}";
                    code += indent + "x = " + expr + ";\n";
                }
            }
            code += indent + "res.push_back(x);\n";
            for (int i=0; i<open_braces; i++) {
                indent = indent.substr(0, indent.size()-4);
                code += indent + "}\n";
            }
            code += "    }\n";
        } else {
            code += "    std::vector<int> res = input;\n";
        }

        for (auto& op : non_fusable) {
            if (op.second == "Sort") code += "    std::sort(res.begin(), res.end());\n";
            else if (op.second == "Reverse") code += "    std::reverse(res.begin(), res.end());\n";
            else if (op.second == "CumulativeSum") code += "    for (size_t i = 1; i < res.size(); ++i) res[i] += res[i-1];\n";
        }
        code += "    return res;\n}\n";
        return code;
    }

    struct SearchNode {
        float f_score;
        float g_score;
        int tiebreaker;
        std::shared_ptr<ASTNode> tree;
        std::vector<State> current_states;

        bool operator<(const SearchNode& o) const {
            if (f_score != o.f_score) return f_score > o.f_score; // Min-heap
            return tiebreaker > o.tiebreaker;
        }
    };

    SynthesisResult synthesize(const std::vector<std::pair<State, State>>& io_examples, int max_steps = 2000) {
        std::vector<State> start_states, goal_states;
        for (auto& io : io_examples) {
            start_states.push_back(io.first);
            goal_states.push_back(io.second);
        }

        std::priority_queue<SearchNode> pq;
        int tiebreaker = 0;
        
        float h_start = heuristic(start_states, goal_states);
        if (h_start == 0.0f) {
            auto input_tree = std::make_shared<ASTNode>("Input", "", nullptr);
            return {input_tree, "def solve(input_list):\n    return input_list", "", render_cpp(input_tree), 1};
        }

        auto initial_tree = std::make_shared<ASTNode>("Input", "", nullptr);
        pq.push({h_start, 0.0f, tiebreaker++, initial_tree, start_states});

        std::set<std::string> visited;
        int steps = 0;

        while (!pq.empty() && steps < max_steps) {
            auto node = pq.top(); pq.pop();
            steps++;

            if (node.current_states == goal_states) {
                return {
                    node.tree,
                    "def solve(input_list):\n    return " + render_python(node.tree),
                    render_java(node.tree),
                    render_cpp(node.tree),
                    steps
                };
            }

            std::string signature = "";
            for (auto& s : node.current_states) signature += s.signature() + "|";
            if (visited.count(signature)) continue;
            visited.insert(signature);

            float new_g = node.g_score + 1.0f;

            auto expand = [&](const std::string& op_type, const std::string& op_name) {
                std::vector<State> new_states;
                bool valid = true;
                for (auto& s : node.current_states) {
                    State ns = apply_op(op_type, op_name, s);
                    if (ns.type == State::EMPTY) { valid = false; break; }
                    new_states.push_back(ns);
                }
                if (valid) {
                    float h = heuristic(new_states, goal_states);
                    if (h != INFINITY) {
                        auto new_tree = std::make_shared<ASTNode>(op_type, op_name, node.tree);
                        pq.push({new_g + h, new_g, tiebreaker++, new_tree, new_states});
                    }
                }
            };

            for (auto& n : map_ops_names) expand("Map", n);
            for (auto& n : filter_ops_names) expand("Filter", n);
            for (auto& n : list_ops_names) expand("List", n);
            for (auto& n : terminal_ops_names) expand("Terminal", n);
        }

        return {nullptr, "", "", "", steps};
    }

    std::string render_java(std::shared_ptr<ASTNode> tree) {
        if (!tree || tree->op_type == "Input") {
            return "public static List<Integer> solve(List<Integer> input) {\n    return input;\n}";
        }
        auto ops = flatten_tree(tree);
        std::string code = "public static List<Integer> solve(List<Integer> input) {\n";
        code += "    List<Integer> res = new ArrayList<>();\n";
        code += "    for (int x : input) {\n";
        std::string indent = "        ";
        int open_braces = 0;
        for (auto& op : ops) {
            if (op.first == "Filter") {
                std::string cond = op.second;
                if (cond == ">0") cond = "x > 0";
                else if (cond == "<0") cond = "x < 0";
                else if (cond == "even") cond = "x % 2 == 0";
                else if (cond == "odd") cond = "x % 2 != 0";
                code += indent + "if (" + cond + ") {\n";
                indent += "    ";
                open_braces++;
            } else if (op.first == "Map") {
                std::string expr = op.second;
                if (expr == "*2") expr = "x * 2";
                else if (expr == "+1") expr = "x + 1";
                else if (expr == "-1") expr = "x - 1";
                else if (expr == "^2") expr = "x * x";
                else if (expr == "abs") expr = "Math.abs(x)";
                code += indent + "x = " + expr + ";\n";
            }
        }
        code += indent + "res.add(x);\n";
        for (int i = 0; i < open_braces; i++) {
            indent = indent.substr(0, indent.size() - 4);
            code += indent + "}\n";
        }
        code += "    }\n";
        for (auto& op : ops) {
            if (op.first == "List" && op.second == "Reverse") {
                code += "    Collections.reverse(res);\n";
            } else if (op.first == "List" && op.second == "Sort") {
                code += "    Collections.sort(res);\n";
            } else if (op.first == "List" && op.second == "CumulativeSum") {
                code += "    for (int i = 1; i < res.size(); i++) res.set(i, res.get(i) + res.get(i - 1));\n";
            }
        }
        code += "    return res;\n}";
        return code;
    }

    static std::vector<int> parse_int_list(const std::string& str) {
        std::vector<int> res;
        std::string num_str;
        for (char c : str) {
            if (std::isdigit(c) || c == '-') {
                num_str += c;
            } else {
                if (!num_str.empty()) {
                    try { res.push_back(std::stoi(num_str)); } catch (...) {}
                    num_str.clear();
                }
            }
        }
        if (!num_str.empty()) {
            try { res.push_back(std::stoi(num_str)); } catch (...) {}
        }
        return res;
    }

    SynthesisResult synthesize_spec(const std::string& spec_str) {
        std::vector<std::pair<State, State>> io_examples;
        std::stringstream ss(spec_str);
        std::string line;
        
        while (std::getline(ss, line, ';')) {
            size_t arrow = line.find("->");
            if (arrow != std::string::npos) {
                std::string lhs = line.substr(0, arrow);
                std::string rhs = line.substr(arrow + 2);
                io_examples.push_back({State(parse_int_list(lhs)), State(parse_int_list(rhs))});
            }
        }

        if (io_examples.empty()) {
            size_t arrow = spec_str.find("->");
            if (arrow != std::string::npos) {
                std::string lhs = spec_str.substr(0, arrow);
                std::string rhs = spec_str.substr(arrow + 2);
                io_examples.push_back({State(parse_int_list(lhs)), State(parse_int_list(rhs))});
            }
        }

        if (io_examples.empty()) {
            return {nullptr, "", "", "", 0};
        }

        auto res = synthesize(io_examples);
        res.code_java = render_java(res.tree);
        return res;
    }
};

} // namespace brain2
