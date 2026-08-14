/*
 * demo_brain2_dataset.cpp — Trains Brain3 on the exact dataset used for Brain2.
 * This reads test_700.txt, extracts the mathematical problems, and feeds them
 * through the Brain3 Cognitive Bridge.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "cognitive_bridge.hpp"

using namespace brain3;
using namespace brain3::engines::synthesis;

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: LEGACY DATASET TRAINING DEMO\n";
    std::cout << "  Training the Fuzzy Core on Brain2's test_700.txt\n";
    std::cout << "=============================================================\n\n";

    // 1. Initialize the Brain
    brain2::Brain brain(32, 32, 64, 128, 7, 500, 8, 42);
    CognitiveBridge bridge(brain);
    bridge.set_log_level(BrainLog::WARN);

    // 2. Load the Brain2 dataset
    std::vector<Problem> curriculum;
    std::ifstream file("../brain2/test_700.txt");
    
    if (!file.is_open()) {
        std::cerr << "Failed to open ../brain2/test_700.txt\n";
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Only grab the algebra problems for now, since Brain3 UnifiedProposer
        // has a native engine for them.
        if (line.find("[algebra]") != std::string::npos) {
            std::string eq = line.substr(10); // strip "[algebra] "
            
            // Format from brain2: "8 x + 42 = 170"
            // Brain3 math parser requires explicit multiplication: "8*x + 42 = 170"
            std::string formatted_eq = "";
            for (size_t i = 0; i < eq.size(); ++i) {
                if (eq[i] == 'x' && i > 0 && eq[i-1] == ' ') {
                    formatted_eq += "*x";
                } else if (eq[i] == ' ') {
                    // Skip spaces before x to ensure number*x
                    if (i + 1 < eq.size() && eq[i+1] == 'x') continue;
                    formatted_eq += eq[i];
                } else {
                    formatted_eq += eq[i];
                }
            }

            Problem p;
            p.type = "equation";
            p.data_str = formatted_eq;
            
            // Split into lhs and rhs for the parser
            size_t eq_pos = formatted_eq.find('=');
            if (eq_pos != std::string::npos) {
                p.lhs = formatted_eq.substr(0, eq_pos);
                p.rhs = formatted_eq.substr(eq_pos + 1);
            }
            
            curriculum.push_back(p);
        }
    }
    file.close();

    std::cout << "Successfully extracted " << curriculum.size() << " algebra problems from Brain2 dataset.\n";
    if (curriculum.empty()) return 1;
    
    std::cout << "First problem: " << curriculum[0].data_str << "\n";
    std::cout << "Starting massive batch training loop...\n\n";

    // 3. Train
    auto start_time = std::chrono::high_resolution_clock::now();
    auto stats = bridge.train_batch(curriculum);
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    // 4. Print results
    std::cout << "=============================================================\n";
    std::cout << "  LEGACY DATASET TRAINING COMPLETE (" << std::fixed << std::setprecision(2) << duration << " seconds)\n";
    std::cout << "=============================================================\n";
    std::cout << "Total problems seen:  " << stats.total << "\n";
    std::cout << "Total solved safely:  " << stats.solved << "\n";
    std::cout << "Episodes stored:      " << stats.episodic_stored 
              << " (highly surprising events)\n";
    std::cout << "Daydreams triggered:  " << stats.daydreams_triggered 
              << " (offline memory consolidations)\n";
    
    std::cout << "\n[Proposer Final Routing Confidence]\n";
    bridge.get_proposer().print_routing_report();

    return 0;
}
