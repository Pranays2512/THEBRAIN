#include <iostream>
#include <iomanip>
#include "reasoning_engine.hpp"
#include "tree_reason.hpp"

using namespace brain2::reasoning;

struct IntState {
    int val;
    bool operator==(const IntState& o) const { return val == o.val; }
    bool operator<(const IntState& o)  const { return val < o.val;  }
};
namespace std {
    template<> struct hash<IntState> {
        size_t operator()(const IntState& s) const { return hash<int>()(s.val); }
    };
}

class SimpleMathProblem : public SearchProblem<IntState> {
public:
    IntState initial() const override { return {1}; }
    bool is_goal(const IntState& s) const override { return s.val == 10; }
    std::vector<std::tuple<std::string, IntState, double>> moves(const IntState& s) const override {
        std::vector<std::tuple<std::string, IntState, double>> m;
        m.push_back({"add 3", {s.val + 3}, 1.0});
        m.push_back({"multiply 2", {s.val * 2}, 1.0});
        return m;
    }
};

void test_reasoning_engine() {
    std::cout << "=== reasoning_engine — symbolic rules and logic ===\n\n";
    ReasoningEngine re;
    re.learn("tom", "parent", "sam");
    re.learn("sam", "parent", "kid");
    re.add_rule("parent", "parent", "grandparent");
    
    auto ans = re.ask("tom", "grandparent");
    std::cout << "tom grandparent ? -> " << ans.first << "\n";
    std::cout << "  because: " << ans.second << "\n";
}

void test_tree_reason() {
    std::cout << "\n=== tree_reason — generalized A* search ===\n\n";
    SimpleMathProblem prob;
    auto result = solve_astar(prob);
    
    std::cout << "Goal: Reach 10 from 1 using (+3) or (*2)\n";
    if (result.solved) {
        std::cout << "  Solved in " << result.path.size() << " steps (expanded " << result.nodes_expanded << " nodes):\n";
        int curr = 1;
        for (const auto& step : result.path) {
            std::cout << "    " << step.first << " -> " << step.second.val << "\n";
        }
    } else {
        std::cout << "  Failed to solve.\n";
    }
}

int main() {
    test_reasoning_engine();
    test_tree_reason();
    return 0;
}
