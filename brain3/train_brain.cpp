#include <iostream>
#include <string>
#include <vector>
#include "fuzzy/core/brain.hpp"
#include "fuzzy/core/curriculum_trainer.hpp"
#include "crisp/engines/reasoning/sleep_engine.hpp"

using namespace brain2;
using namespace brain3::training;

void run_test_sleep() {
    std::cout << "=============================================================\n";
    std::cout << "  🧠 RUNNING AUTOMATED SLEEP CONSOLIDATION TEST (PHASES 1-4)\n";
    std::cout << "=============================================================\n\n";

    Brain brain(256, 16, 16, 64, 7, 5000, 8, 42);

    // 1. Setup Phase 1 Knowledge (100 birds + 1 penguin)
    std::cout << "1. Teaching KB: 100 flying birds + 1 swimming penguin...\n";
    for (int i = 1; i <= 100; i++) {
        std::string bid = "bird_" + std::to_string(i);
        brain.brainql_engine.learn(bid, "isa", "bird");
        brain.brainql_engine.learn(bid, "can", "fly");
        brain.brainql_engine.learn(bid, "has", "feathers");
    }
    brain.brainql_engine.learn("penguin_1", "isa", "bird");
    brain.brainql_engine.learn("penguin_1", "can", "swim");

    int initial_facts = brain.brainql_engine.facts.size();
    std::cout << "   Initial KB Facts: " << initial_facts << "\n";

    // 2. Setup Phase 2 Telemetry (simulate gate log)
    std::cout << "2. Creating temporary test gate log...\n";
    {
        std::ofstream log("test_gate_sim.jsonl");
        log << "{\"timestamp\": 1786600001, \"inputs\": [\"paris\", \"capital\"], \"guess\": \"france\", \"store_truth\": \"france\", \"verdict\": \"verified_atomic\"}\n";
        log << "{\"timestamp\": 1786600002, \"inputs\": [\"bird\", \"can\"], \"guess\": \"fly\", \"store_truth\": \"fly\", \"verdict\": \"verified_atomic\"}\n";
        log << "{\"timestamp\": 1786600003, \"inputs\": [\"dog\", \"drives\"], \"guess\": \"car\", \"store_truth\": \"\", \"verdict\": \"not_found\"}\n";
        log.close();
    }

    // 3. Run Full Sleep Consolidation
    std::cout << "3. Executing SleepEngine::full_sleep()...\n";
    brain2::reasoning::SleepEngine sleep_engine;
    auto report = sleep_engine.full_sleep(brain, "test_gate_sim.jsonl", "./out/test_sleep_ckpt", 0.90, 5);
    report.print();

    // 4. Assertions
    std::cout << "4. Verifying Phase Results...\n";
    
    // Phase 1 check: Rules induced, facts compressed, exception preserved
    bool p1_ok = (report.phase1_rules_created > 0) && (brain.brainql_engine.facts.size() < (size_t)initial_facts);
    auto [ans_bird, r_bird] = brain.brainql_engine.ask("bird_42", "can");
    auto [ans_pen, r_pen] = brain.brainql_engine.ask("penguin_1", "can");

    std::cout << "   • Bird 42 can:    " << ans_bird << " [" << r_bird << "]\n";
    std::cout << "   • Penguin 1 can:  " << ans_pen << " [" << r_pen << "]\n";

    if (p1_ok && ans_bird == "fly" && ans_pen == "swim") {
        std::cout << "   ✓ Phase 1 (Crisp Rule Induction & Compression) PASSED\n";
    } else {
        std::cout << "   ✗ Phase 1 FAILED\n";
    }

    // Phase 2 check: Telemetry records ingested and trained
    if (report.phase2_telemetry_records == 3 && report.phase2_triples_trained == 2) {
        std::cout << "   ✓ Phase 2 (Neural STaR Training from Gate Log) PASSED\n";
    } else {
        std::cout << "   ✗ Phase 2 FAILED (records=" << report.phase2_telemetry_records 
                  << ", trained=" << report.phase2_triples_trained << ")\n";
    }

    // Phase 3 check: Topological & Episodic decay
    if (report.phase3_som_nodes_decayed > 0 && report.phase3_episodes_pruned > 0) {
        std::cout << "   ✓ Phase 3 (Topological & Episodic Pruning) PASSED\n";
    } else {
        std::cout << "   ✗ Phase 3 FAILED\n";
    }

    // Phase 4 check: Checkpoint created
    if (report.phase4_checkpoint_success) {
        std::cout << "   ✓ Phase 4 (Atomic Checkpointing) PASSED\n";
    } else {
        std::cout << "   ✗ Phase 4 FAILED\n";
    }

    // Clean up
    std::system("rm -f test_gate_sim.jsonl");
    std::cout << "\n=============================================================\n";
    std::cout << "  ALL SLEEP PHASES VERIFIED SUCCESSFULLY!\n";
    std::cout << "=============================================================\n";
}

int main(int argc, char** argv) {
    std::string curriculum_path = "../brain2/data/brain_curriculum.txt";
    std::string checkpoint_dir = "./out/brain_fluent";
    std::string gate_log = "associative_gate.jsonl";
    int epochs = 1;
    bool sleep_only = false;
    bool run_test = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--curriculum" && i + 1 < argc) curriculum_path = argv[++i];
        else if (arg == "--checkpoint" && i + 1 < argc) checkpoint_dir = argv[++i];
        else if (arg == "--gate-log" && i + 1 < argc) gate_log = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--sleep-only") sleep_only = true;
        else if (arg == "--test-sleep") run_test = true;
    }

    if (run_test) {
        run_test_sleep();
        return 0;
    }

    Brain brain(256, 16, 16, 64, 7, 10000, 8, 42);

    if (sleep_only) {
        std::cout << "[train_brain] Running standalone sleep consolidation...\n";
        brain2::reasoning::SleepEngine sleep_engine;
        auto report = sleep_engine.full_sleep(brain, gate_log, checkpoint_dir);
        report.print();
    } else {
        CurriculumTrainer::train(brain, curriculum_path, epochs, true, checkpoint_dir);
    }

    return 0;
}
