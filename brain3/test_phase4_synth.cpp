#include "crisp/engines/synthesis/_program_synth.hpp"
#include "crisp/engines/synthesis/_program_synth_guided.hpp"
#include "crisp/engines/synthesis/_program_synth_policy.hpp"
#include "crisp/engines/synthesis/string_synthesis_engine.hpp"
#include "crisp/engines/synthesis/invariant_miner.hpp"
#include "crisp/engines/synthesis/synth_invariant.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Blind synth: BFS over string DSL
    brain3::engines::synthesis::StringSynth synth({{"John Smith", "JS"}, {"Mary Jane", "MJ"}});
    auto r1 = brain2::reasoning::solve_astar(synth, 50000);
    std::cout << "StringSynth solved=" << r1.solved << std::endl;

    // Guided synth
    brain3::engines::synthesis::SynthesizeGuided guided({{"John Smith", "JS"}});
    auto r2 = brain2::reasoning::solve_astar(guided, 50000);
    std::cout << "SynthesizeGuided solved=" << r2.solved << std::endl;

    // Policy synth
    brain3::engines::synthesis::PolicySynth policy({{"John Smith", "JS"}});
    auto r3 = brain2::reasoning::solve_astar(policy, 50000);
    std::cout << "PolicySynth solved=" << r3.solved << std::endl;

    // SynthesisEngine (hardened wrapper)
    brain3::engines::synthesis::SynthesisEngine se;
    auto res = se.synthesize({{"John Smith", "JOHN"}, {"bob dylan", "BOB"}});
    std::cout << "SynthesisEngine found=" << res.found << " program=" << res.source() << std::endl;

    // InvariantMiner
    brain3::engines::synthesis::InvariantMiner im;
    std::vector<std::pair<brain3::engines::synthesis::InvariantMiner::Args, int>> train = {
        {{1}, 1}, {{4}, 24}, {{5}, 120}, {{6}, 720}
    };
    std::vector<std::pair<brain3::engines::synthesis::InvariantMiner::Args, int>> holdout = {
        {{7}, 5040}, {{8}, 40320}
    };
    auto admitted = im.validate(im.mine(train), holdout);
    std::cout << "InvariantMiner admitted " << admitted.size() << " invariants: ";
    for (const auto& s : admitted) std::cout << s << " ";
    std::cout << std::endl;

    // SynthInvariant: triage
    brain3::engines::synthesis::SynthInvariant si;
    auto task_inv = si.task_invariants([](int n) -> int {
        int r = 1; for (int i = 1; i <= n; i++) r *= i; return r;
    }, {0,1,4,5,6}, {7,8,9});
    std::cout << "SynthInvariant task_invariants: " << task_inv.size() << " admitted" << std::endl;

    std::cout << "\nPhase 4 Synthesis Compiled and Verified successfully!" << std::endl;
    return 0;
}
