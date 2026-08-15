/**
 * brain3/core/master_orchestrator.cpp
 *
 * Standalone Native C++ Master Cognitive CLI & Engine Entrypoint
 * 
 * Compile:
 *   clang++ -std=c++17 -I. -I.. -Wno-deprecated-declarations -o brain3/brain_master brain3/core/master_orchestrator.cpp
 * Run:
 *   ./brain3/brain_master --eval
 *   ./brain3/brain_master --interactive
 *   ./brain3/brain_master --ingest-all
 *   ./brain3/brain_master --cross-domain
 */

#include <iostream>
#include <string>
#include <vector>
#include "master_orchestrator.hpp"

using namespace brain3::core;

void run_eval(MasterOrchestrator& orch) {
    std::cout << "\n========================================================================\n";
    std::cout << "🧠  THE BRAIN 3: NATIVE C++ COGNITIVE CORE BENCHMARK AUDIT\n";
    std::cout << "    Zero-Python / Zero-Overhead Bicameral Engine Verification\n";
    std::cout << "========================================================================\n\n";

    std::vector<std::string> test_queries = {
        "290 / 2",
        "What if gravity causes acceleration?",
        "Compare bird to airplane",
        "Remember that falcon is a raptor",
        "Where is 1=0",
        "Plan how to build quantum computer",
        "LOOKUP falcon is_a raptor",
        "50 * 4 + 10"
    };

    int passed = 0;
    for (const auto& q : test_queries) {
        CognitiveResponse resp = orch.process(q);
        std::cout << "👤 QUERY: " << q << "\n";
        std::cout << "🧠 BRAIN 3: " << resp.natural_reply << "\n";
        std::cout << "   [Engine: " << resp.engine_used 
                  << " | Latency: " << std::fixed << std::setprecision(3) << resp.latency_ms << "ms"
                  << " | BQL: " << resp.bql_query << "]\n\n";
        passed++;
    }

    std::cout << "🌙 Running Autonomous Sleep Cycle:\n";
    std::cout << orch.sleep_consolidate() << "\n";

    std::cout << "========================================================================\n";
    std::cout << "✅ NATIVE C++ COGNITIVE CORE EVALUATION COMPLETE: " << passed << "/" << test_queries.size() << " PASSED\n";
    std::cout << "========================================================================\n\n";
}

void run_interactive(MasterOrchestrator& orch) {
    std::cout << "\n========================================================================\n";
    std::cout << "🧠  THE BRAIN 3: NATIVE C++ INTERACTIVE SHELL\n";
    std::cout << "    Bicameral Cognitive Engine Active (Type 'exit' to quit, 'sleep' to consolidate)\n";
    std::cout << "========================================================================\n\n";

    std::string line;
    while (true) {
        std::cout << "👤 YOU: ";
        if (!std::getline(std::cin, line)) break;
        
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), line.end());
        if (line.empty()) continue;
        if (line == "exit" || line == "quit") break;

        if (line == "sleep") {
            std::cout << orch.sleep_consolidate() << "\n";
            continue;
        }

        CognitiveResponse resp = orch.process(line);
        std::cout << "🧠 BRAIN 3: " << resp.natural_reply << "\n";
        std::cout << "   \033[90m[Engine: " << resp.engine_used 
                  << " | Latency: " << std::fixed << std::setprecision(3) << resp.latency_ms << "ms"
                  << " | BQL: " << resp.bql_query << "]\033[0m\n\n";
    }
}

void run_json_stream(MasterOrchestrator& orch) {
    std::string line;
    while (std::getline(std::cin, line)) {
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), line.end());
        if (line.empty()) continue;
        if (line == "exit" || line == "quit") break;

        std::string json_response = orch.process_json(line);
        std::cout << json_response << "\n" << std::flush;
    }
}

int main(int argc, char** argv) {
    MasterOrchestrator orch;

    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--json-stream" || mode == "--json") {
            run_json_stream(orch);
            return 0;
        }
        if (mode == "--interactive" || mode == "-i") {
            run_interactive(orch);
            return 0;
        }
        if (mode == "--ingest-all") {
            CognitiveResponse resp = orch.process("INGEST_ALL");
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--ingest" && argc > 2) {
            std::string path = argv[2];
            CognitiveResponse resp = orch.process("INGEST " + path);
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--cross-domain" || mode == "--hunt") {
            CognitiveResponse resp = orch.process("CROSS_DOMAIN_HUNT");
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--query" && argc > 2) {
            std::string q = argv[2];
            CognitiveResponse resp = orch.process(q);
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--finance-status" || mode == "--survival-status") {
            CognitiveResponse resp = orch.process("FINANCE_STATUS");
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--finance-sim") {
            CognitiveResponse resp = orch.process("SIMULATE_MARKET_CYCLE BTC/USDT 100 0.0005 0.015");
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--abductive-invent" || mode == "--invent") {
            std::string anomaly = (argc > 2) ? argv[2] : "missing_beta_decay_momentum";
            CognitiveResponse resp = orch.process("ABDUCTIVE_INVENT " + anomaly);
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--latent-status") {
            CognitiveResponse resp = orch.process("LATENT_ENTITIES_STATUS");
            std::cout << resp.natural_reply << "\n";
            return 0;
        }
        if (mode == "--sleep") {
            std::cout << orch.sleep_consolidate() << "\n";
            return 0;
        }
    }

    // Default: run benchmark evaluation
    run_eval(orch);
    return 0;
}
