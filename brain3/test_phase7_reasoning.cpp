#include "crisp/engines/reasoning/tree_domains.hpp"
#include "crisp/engines/reasoning/tree_learn.hpp"
#include "crisp/engines/reasoning/learned_guidance.hpp"
#include "crisp/engines/reasoning/deeper_grammar.hpp"
#include "crisp/engines/reasoning/nested_parser.hpp"
#include "crisp/engines/reasoning/structural_parser.hpp"
#include "crisp/engines/reasoning/planning_engine.hpp"
#include "crisp/engines/reasoning/brain_planner.hpp"
#include "crisp/engines/reasoning/means_ends.hpp"
#include "crisp/engines/reasoning/monte_carlo_tree.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace brain2::reasoning;

class NoveltyLineProblem : public SearchProblem<int> {
public:
    int initial() const override { return 1; }
    bool is_goal(const int& s) const override { return s == 22; }
    double heuristic(const int& s) const override { return std::abs(22 - s); }
    double novelty(const int& s) const override {
        return s >= 1 && s <= 22 ? static_cast<double>(s) / 22.0 : 0.0;
    }
    std::vector<std::tuple<std::string, int, double>> moves(const int& s) const override {
        std::vector<std::tuple<std::string, int, double>> m;
        if (s < 25) {
            m.push_back({"add 1", s + 1, 1.0});
            m.push_back({"multiply 2", s * 2, 1.0});
        }
        return m;
    }
};

int main() {
    // 1. tree_domains
    NQueens q(4);
    assert(solve_astar(q).path.size() > 0);
    std::cout << "TreeDomains OK\n";

    // 2. tree_learn
    auto start = scramble_8puzzle(4);
    EightPuzzle ep(start);
    assert(solve_astar(ep).path.size() > 0);
    std::cout << "TreeLearn OK\n";

    // 3. learned_guidance
    LearnedHeuristic lh(eight_puzzle_features);
    auto ex = collect_8puzzle_examples(5, 4, 1);
    lh.train(ex);
    std::cout << "LearnedGuidance OK\n";

    // 4. means_ends
    ReasoningEngine kb;
    kb.learn("rocket", "mass", "1000");
    kb.learn("rocket", "accel", "20");
    kb.learn("rocket", "speed", "300");

    PolicyMemory mem;
    mem.add({"force", {"mass", "accel"}, op("*", {var("mass"), var("accel")})});
    mem.add({"power", {"force", "speed"}, op("*", {var("force"), var("speed")})});

    FactSource fs(&kb);
    PolicySource ps(&mem);
    MeansEndsSolver solver({&fs, &ps});

    auto p = solver.solve(Need{"rocket", "power"});
    assert(p && *p == 6000000);
    std::cout << "MeansEnds OK\n";

    // 5. deeper_grammar
    DeeperParser dp({&fs, &ps});
    assert(dp.answer("if the rocket mass is greater than 500 and its speed is greater than 100 then what is its force") != "(abstain)");
    std::cout << "DeeperGrammar OK\n";

    // 6. nested_parser
    NestedParser np({&fs, &ps});
    assert(np.answer("what is the force of the rocket").find("force") != std::string::npos);
    std::cout << "NestedParser OK\n";

    // 7. structural_parser
    StructuralParser sp({"rocket", "sample"}, {"force", "density", "mass", "speed", "accel", "volume"});
    auto query = sp.parse("what is the force of the rocket");
    assert(query.kind == "single" && query.q_entities[0] == "rocket" && query.q_relations[0] == "force");
    std::cout << "StructuralParser OK\n";

    // 8. planning_engine
    PlanningEngine pe;
    pe.define_action("smelt", {"ore"}, {"iron"});
    pe.define_action("chop", {"axe"}, {"wood"});
    pe.define_action("forge", {"iron", "wood"}, {"sword"});
    auto plan = pe.plan({"ore", "axe"}, "sword");
    assert(plan.found);
    std::cout << "PlanningEngine OK\n";

    // 9. brain_planner
    FactWorld fw;
    fw.teach("smelt", "requires", "ore");
    fw.teach("smelt", "produces", "iron");
    CraftPlanBrain cpb(&fw, {"ore"}, "iron");
    assert(solve_astar(cpb).path.size() > 0);
    std::cout << "BrainPlanner OK\n";

    // 10. monte_carlo_tree
    NoveltyLineProblem novelty_prob;
    MonteCarloConfig cfg;
    cfg.iterations = 300;
    cfg.rollout_depth = 8;
    cfg.seed = 11;
    auto mc = solve_mcts(novelty_prob, cfg);
    assert(mc.solved);
    assert(!mc.path.empty());
    assert(mc.path.back().second == 22);
    std::cout << "MonteCarloTree OK\n";

    std::cout << "Phase 7 compiled and verified successfully!\n";
    return 0;
}
