#include <iostream>
#include <iomanip>
#include "appraisal_engine.hpp"
#include "conversation_engine.hpp"

using namespace brain2::faculties;

int main() {
    std::cout << "=== faculties test (crisp hemisphere) ===\n\n";
    
    // Test 1: Appraisal Engine (Pragmatics parsing)
    AppraisalEngine ae;
    std::vector<std::string> inputs = {
        "hey, how are you?", 
        "what is apple?", 
        "tell me about you", 
        "the apple is red"
    };
    
    std::cout << "Appraisal Engine Results:\n";
    for (const auto& text : inputs) {
        auto ap = ae.appraise(text);
        std::cout << "  \"" << text << "\"\n";
        std::cout << "    type: " << ap.type << "\n    dims: ";
        for (const auto& [dim, val] : ap.frame) {
            if (val > 0.1) std::cout << dim << ":" << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << "\n\n";
    }
    
    // Test 2: Conversation Engine
    std::cout << "Conversation Engine Results:\n";
    ConversationEngine ce;
    std::cout << "  > hello\n    " << ce.respond("hello") << "\n";
    std::cout << "  > what is apple?\n    " << ce.respond("what is apple?") << "\n";
    
    return 0;
}
