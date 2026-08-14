#pragma once
#include <string>

namespace brain2 {
namespace routing {

enum class CrispMode {
    IDLE,
    RETRIEVE,
    VERIFY,
    SYNTHESIZE,
    PROPOSE,
    CURIOUS
};

struct CrispRoutingDecision {
    CrispMode mode;
    std::string label;
    bool trigger_teach = false;
    bool trigger_propose = false;
    bool trigger_curiosity = false;
    double confidence_out = 0.0;
};

// The CrispInternalRouter is the mode-switcher for the crisp layer,
// deciding what action the engine should take based on cognitive signals.
class CrispInternalRouter {
public:
    CrispRoutingDecision decide(
        double confidence = 0.0,
        int verification_depth = 0,
        double novelty = 0.0,
        double curiosity_error = 0.0,
        std::string solution_type = "none",
        std::string appraisal_type = "statement",
        bool is_verified = false
    ) {
        CrispRoutingDecision d;
        d.mode = CrispMode::IDLE;
        
        if (appraisal_type == "greeting") {
            d.mode = CrispMode::IDLE;
            d.label = "IDLE(social)";
            d.confidence_out = 1.0;
            return d;
        }
        
        if (is_verified && confidence >= 0.85 && verification_depth <= 1) {
            d.mode = CrispMode::RETRIEVE;
            d.label = "RETRIEVE(high_conf)";
            d.trigger_teach = true;
            d.confidence_out = confidence;
            return d;
        }
        
        if (curiosity_error >= 0.70 || novelty >= 0.65) {
            d.mode = CrispMode::PROPOSE;
            d.label = "PROPOSE(high_curiosity)";
            d.trigger_propose = true;
            d.trigger_curiosity = true;
            d.confidence_out = 1.0 - novelty;
            return d;
        }
        
        if (solution_type == "compute" || solution_type == "code") {
            d.mode = CrispMode::SYNTHESIZE;
            d.label = "SYNTHESIZE";
            d.confidence_out = 0.0;
            return d;
        }
        
        if (curiosity_error >= 0.35) {
            d.mode = CrispMode::CURIOUS;
            d.label = "CURIOUS";
            d.trigger_curiosity = true;
            d.confidence_out = confidence;
            return d;
        }
        
        d.label = "IDLE(no-signal)";
        return d;
    }
};

} // namespace routing
} // namespace brain2
