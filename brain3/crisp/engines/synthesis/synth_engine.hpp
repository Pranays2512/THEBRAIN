#pragma once
#include "crisp/engines/synthesis/logic_plan.hpp"
#include "crisp/engines/synthesis/online_proposer2.hpp"
#include "crisp/engines/synthesis/policy_proposer.hpp"
#include "crisp/engines/synthesis/conjecture_sandbox.hpp"
#include "crisp/engines/synthesis/refuter.hpp"
#include "crisp/engines/synthesis/refute_synth.hpp"
#include "crisp/engines/synthesis/irregularity_detector.hpp"

// SynthEngine wrapper acts as the unified frontend over all synthesizers
namespace brain3 {
namespace engines {
namespace synthesis {

class SynthEngine {
public:
    // We would wire up all the L1/L2/L3/L4/Graph synth rules here.
    // In Python this was a registry of routes `ROUTES`.
    // In C++, this provides a single entry point for a task kind.
    
    // For Phase 6 port completeness, this header ties together the 
    // unified synthesis ecosystem components.
    
    SynthEngine() = default;
};

}}}
