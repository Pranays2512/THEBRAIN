/*
 * demo_batch_training.cpp — Trains the Brain on a batch of problems to
 * demonstrate continuous learning, emotion modulation, episodic memory
 * accumulation, and intuition adjustment.
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include "cognitive_bridge.hpp"

using namespace brain3;
using namespace brain3::engines::synthesis;

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: BATCH TRAINING DEMO\n";
    std::cout << "  Training the Fuzzy Core and Intuition Router\n";
    std::cout << "=============================================================\n\n";

    // 1. Initialize the Brain (small for fast execution)
    brain2::Brain brain(32, 32, 64, 128, 7, 500, 8, 42);
    CognitiveBridge bridge(brain);
    
    // Set logging to WARNING only so the console isn't flooded,
    // we just want to see the final stats.
    bridge.set_log_level(BrainLog::WARN);

    // 2. Generate a curriculum of problems
    std::vector<Problem> curriculum;
    
    // Algebra block
    for (int i = 1; i <= 10; ++i) {
        Problem p;
        p.type = "equation";
        p.data_str = std::to_string(i) + "x + " + std::to_string(i*2) + " = " + std::to_string(i*5);
        p.lhs = std::to_string(i) + "*x + " + std::to_string(i*2);
        p.rhs = std::to_string(i*5);
        curriculum.push_back(p);
    }
    
    // Calculus block
    for (int i = 1; i <= 10; ++i) {
        Problem p;
        p.type = "integrate";
        p.data_str = std::to_string(i) + "*x^" + std::to_string(i);
        p.expr_str = p.data_str;
        curriculum.push_back(p);
    }

    std::cout << "Curriculum loaded: " << curriculum.size() << " problems.\n";
    std::cout << "Starting batch training loop...\n\n";

    // 3. Train
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto stats = bridge.train_batch(curriculum);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    // 4. Print results
    std::cout << "=============================================================\n";
    std::cout << "  TRAINING COMPLETE (" << std::fixed << std::setprecision(2) << duration << " seconds)\n";
    std::cout << "=============================================================\n";
    std::cout << "Total problems seen:  " << stats.total << "\n";
    std::cout << "Total solved safely:  " << stats.solved << "\n";
    std::cout << "Episodes stored:      " << stats.episodic_stored 
              << " (highly surprising events)\n";
    std::cout << "Daydreams triggered:  " << stats.daydreams_triggered 
              << " (offline memory consolidations)\n";
    std::cout << "Avg Prediction Error: " << std::fixed << std::setprecision(4) 
              << stats.avg_prediction_error << "\n";
    
    std::cout << "\n[Brain Final State]\n";
    bridge.print_state();
    
    std::cout << "\n[Proposer Final Routing Confidence]\n";
    bridge.get_proposer().print_routing_report();

    // 5. Save the trained brain
    try {
        std::system("mkdir -p ./out/brain_trained");
        bridge.save("./out/brain_trained");
        std::cout << "\n[Save] ✓ Fully trained brain saved to ./out/brain_trained/\n";
    } catch (const std::exception& e) {
        std::cout << "\n[Save] Warning: " << e.what() << "\n";
    }

    return 0;
}
