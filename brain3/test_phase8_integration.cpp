#include "crisp/engines/reasoning/dual_process_engine.hpp"
#include "crisp/engines/reasoning/brainql.hpp"
#include "crisp/engines/reasoning/neuro_bridge.hpp"
#include "crisp/engines/reasoning/relational_query.hpp"
#include "crisp/engines/reasoning/tree_domains.hpp"
#include <iostream>
#include <cassert>

using namespace brain2::reasoning;

int main() {
    std::cout << "Starting Phase 8 tests...\n";

    // 1. DualProcessEngine
    NQueens q(4);
    DualProcessSolver<std::vector<int>> dps(4);
    auto dps_res = dps.solve(q);
    assert(dps_res.found);
    assert(dps_res.tier == "deliberation");
    
    auto dps_res2 = dps.solve(q);
    assert(dps_res2.found);
    assert(dps_res2.tier == "memory");
    std::cout << "DualProcessEngine OK\n";

    // 2. BrainQL
    ReasoningEngine kb;
    BrainQLExecutor bqle(&kb);
    bqle.run(parse_bql("TEACH cat isa animal"));
    bqle.run(parse_bql("TEACH animal is alive"));
    auto bql_res = bqle.run(parse_bql("INHERIT cat is"));
    assert(bql_res.known && bql_res.value == "alive");
    std::cout << "BrainQL OK\n";

    // 3. NeuroBridge
    std::shared_ptr<Eyes> eyes = std::make_shared<RuleEyes>();
    std::shared_ptr<Brain> brain = std::make_shared<Brain>();
    std::shared_ptr<Mouth> mouth = std::make_shared<GrammarMouth>();
    Mind mind(eyes, brain, mouth);
    
    mind.teach("cat", "isa", "animal");
    std::string out = mind.respond("TEACH animal is alive");
    std::string out2 = mind.respond("INHERIT cat is");
    assert(out2.find("alive") != std::string::npos);
    std::cout << "NeuroBridge OK\n";

    // 4. RelationalQuery
    brain->teach("rocket", "mass", "1000");
    brain->teach("probe", "mass", "200");
    PolicyMemory mem;
    RelationalParser rp({"rocket", "probe"});
    std::string rq_ans = rp.answer("is the rocket heavier than the probe?", brain->get_engine(), &mem);
    assert(rq_ans.find("Yes") == 0);
    std::string rq_ans2 = rp.answer("is the probe heavier than the rocket?", brain->get_engine(), &mem);
    assert(rq_ans2.find("No") == 0);
    std::cout << "RelationalQuery OK\n";

    std::cout << "Phase 8 compiled and verified successfully!\n";
    return 0;
}
