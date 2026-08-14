#include "crisp/engines/synthesis/_loop_synth.hpp"
#include "crisp/engines/synthesis/_loop_synth_v2.hpp"
#include "crisp/engines/synthesis/_loop_synth_v3.hpp"
#include "crisp/engines/synthesis/_loop_synth_v4.hpp"
#include "crisp/engines/synthesis/composable_synth.hpp"
#include <iostream>

int main() {
    brain3::engines::synthesis::LoopSynth l1;
    brain3::engines::synthesis::LoopSynthV2 l2;
    brain3::engines::synthesis::LoopSynthV3 l3;
    brain3::engines::synthesis::LoopSynthV4 l4;
    brain3::engines::synthesis::ComposableSynth c;
    
    std::cout << "Phase 2 Synthesis Compiled successfully!" << std::endl;
    return 0;
}
