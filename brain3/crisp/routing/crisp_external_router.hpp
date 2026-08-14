#pragma once
#include <string>
#include <vector>
#include <iostream>

namespace brain2 {
namespace routing {

struct CrispFact {
    std::string entity;
    std::string relation;
    double value;
    bool verified = false;
    std::string source = "crisp_reasoner";
    double confidence = 1.0;
};

struct PushResult {
    bool accepted;
    std::string reason;
};

// The CrispExternalRouter is the outward membrane gating facts and policies 
// verified by the Crisp layer before they cross into the Fuzzy layer.
class CrispExternalRouter {
private:
    std::vector<CrispFact> buffer;
    int facts_accepted = 0;
    int facts_rejected = 0;

public:
    PushResult push_fact(const CrispFact& fact) {
        // Gate 1: Unverified facts NEVER cross
        if (!fact.verified) {
            facts_rejected++;
            return {false, "unverified"};
        }
        
        // Gate 2: Confidence floor
        if (fact.confidence < 0.5) {
            facts_rejected++;
            return {false, "low_confidence"};
        }
        
        // Push across membrane (buffered in this stub implementation, 
        // to be ingested by the Fuzzy Brain module)
        buffer.push_back(fact);
        facts_accepted++;
        return {true, "ok"};
    }
    
    int get_accepted_count() const { return facts_accepted; }
    int get_rejected_count() const { return facts_rejected; }
};

} // namespace routing
} // namespace brain2
