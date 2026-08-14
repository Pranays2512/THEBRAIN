/*
 * demo_native_chat.cpp — Direct Biological Communication
 *
 * Demonstrates having a "normal communication" with the C++ Biological Core
 * using its 400,000-word native language fluency. Bypasses the LLM completely.
 */
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "fuzzy/core/brain.hpp"

using namespace brain2;

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: NATIVE BIOLOGICAL CHAT (NO LLM)\n";
    std::cout << "=============================================================\n\n";

    Brain brain(16, 16, 128, 128, 7, 500, 8, 42);

    std::cout << "[1] Loading Fluent Biological Core (from GloVe ingest)...\n";
    try {
        brain.load_components(
            "./out/brain_fluent/predictor.bin",
            "./out/brain_fluent/language.bin",
            "./out/brain_fluent/som.bin",
            "./out/brain_fluent/episodic.bin",
            "./out/brain_fluent/emotion.bin",
            "./out/brain_fluent/self.bin",
            "./out/brain_fluent/symbolic.bin"
        );
    } catch (...) {
        std::cerr << "Failed to load fluent brain. Did you run demo_glove_import first?\n";
        return 1;
    }
    
    std::cout << "[2] Core online. Brain is ready to communicate natively.\n\n";

    std::cout << "[3] Testing Biological Memory Binding (Learning and Recalling natively)...\n\n";

    std::string input_line;
    while (true) {
        std::cout << "👤 YOU: ";
        if (!std::getline(std::cin, input_line)) break;
        if (input_line.empty()) continue;
        if (input_line == "quit" || input_line == "exit") break;

        // Simple heuristic: if it contains " is_in ", we learn. If it contains "What is_in ", we query.
        auto is_in_pos = input_line.find(" is_in ");
        if (is_in_pos != std::string::npos) {
            std::string subj = input_line.substr(0, is_in_pos);
            std::string obj = input_line.substr(is_in_pos + 7);
            
            auto v_subj = brain.language.encode(subj);
            auto v_isin = brain.language.encode("is_in");
            auto v_obj = brain.language.encode(obj);
            
            brain.binding.bind(v_subj, v_isin, v_obj);
            std::cout << "🧠 BRAIN: [Biological Core committed spatial binding to Hippocampus]\n\n";
        } 
        else if (input_line.find("What is_in ") == 0) {
            std::string subj = input_line.substr(11);
            if (!subj.empty() && subj.back() == '?') subj.pop_back();
            
            auto v_subj = brain.language.encode(subj);
            auto v_isin = brain.language.encode("is_in");
            
            auto recall = brain.binding.query(v_subj, v_isin, true, 0.4f);
            
            std::cout << "🧠 BRAIN THOUGHTS: ";
            if (recall.second == 0.0f) {
                std::cout << "I don't know.\n\n";
            } else {
                auto decoded = brain.language.decode(recall.first, 1);
                std::cout << "My semantic memory recalls: " << decoded[0].first << " (confidence: " << recall.second << ")\n\n";
            }
        }
        else {
            // General word encoding test
            auto v = brain.language.encode(input_line);
            auto decoded = brain.language.decode(v, 1);
            std::cout << "🧠 BRAIN THOUGHTS: " << decoded[0].first << "\n\n";
        }
    }

    std::cout << "\n=============================================================\n";
    std::cout << "  SESSION COMPLETE\n";
    std::cout << "=============================================================\n";
    return 0;
}
