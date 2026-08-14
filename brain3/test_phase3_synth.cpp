#include "crisp/engines/synthesis/graph_synth.hpp"
#include "crisp/engines/synthesis/dp_proposer.hpp"
#include "crisp/engines/synthesis/dp_greedy_synth.hpp"
#include "crisp/engines/synthesis/composable_proposer.hpp"
#include "crisp/engines/synthesis/_program_synth_tree.hpp"
#include <iostream>

int main() {
    brain3::engines::synthesis::GraphSynth g;
    brain3::engines::synthesis::DPProposer dp;
    brain3::engines::synthesis::DPGreedySynth dpg;
    brain3::engines::synthesis::ComposableProposer cp;
    brain3::engines::synthesis::DecisionTree tree;
    
    std::cout << "Phase 3 Synthesis Compiled successfully!" << std::endl;
    return 0;
}
