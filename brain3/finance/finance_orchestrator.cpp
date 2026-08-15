#include "finance_orchestrator.hpp"
#include <iostream>
#include <string>

using namespace brain3::finance;

int main(int argc, char* argv[]) {
    // Exact User Parameters: ₹1,000 Starting Cash, -₹1,000 Ruin Floor, ₹100,000 Cap Limit
    FinanceOrchestrator orch(1000.0, -1000.0, 100000.0, 0.02);

    bool json_stream = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json-stream" || arg == "-j") {
            json_stream = true;
        } else if (arg == "--status") {
            std::cout << orch.execute_command("FINANCE_STATUS") << std::endl;
            return 0;
        } else if (arg == "--sample-trade") {
            std::cout << orch.execute_command("SAMPLE_SURVIVAL_TRADE NIFTY50/INR") << std::endl;
            return 0;
        } else if (arg == "--autonomous-survival" || arg == "--auto") {
            std::cout << orch.execute_command("AUTONOMOUS_SURVIVAL_CYCLE 300") << std::endl;
            return 0;
        } else if (arg == "--sim") {
            std::cout << orch.execute_command("SIMULATE_MARKET_CYCLE BTC/INR 100 0.0005 0.015") << std::endl;
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "THE BRAIN 3: Quantitative Finance Dedicated Branch with Survival Instinct\n"
                      << "Parameters: Initial = 1,000 INR | Ruin Floor = -1,000 INR | Cap = 100,000 INR\n"
                      << "Usage:\n"
                      << "  ./brain_finance --json-stream         # Interactive JSON-stream pipe\n"
                      << "  ./brain_finance --status              # Dump current survival instinct status\n"
                      << "  ./brain_finance --sample-trade        # Execute a live sample survival trade\n"
                      << "  ./brain_finance --autonomous-survival # Run autonomous multi-tick survival loop\n"
                      << "  ./brain_finance --sim                 # Run Monte Carlo market cycle\n";
            return 0;
        }
    }

    if (json_stream) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line == "QUIT" || line == "EXIT") break;
            std::string resp = orch.execute_command(line);
            std::cout << resp << std::endl;
            std::cout.flush();
        }
        return 0;
    }

    // Default interactive / banner mode
    std::cout << "========================================================================\n"
              << "🧠  THE BRAIN 3: QUANTITATIVE FINANCE SURVIVAL INSTINCT BRANCH\n"
              << "    Parameters: Starting ₹1,000 | Ruin Floor -₹1,000 | Cap Limit ₹100,000\n"
              << "========================================================================\n";
    std::cout << "Initial Survival State:\n" << orch.execute_command("FINANCE_STATUS") << "\n\n";

    std::cout << "Executing Sample Survival Trade (NIFTY50/INR)...\n";
    std::cout << orch.execute_command("SAMPLE_SURVIVAL_TRADE NIFTY50/INR") << "\n\n";

    std::cout << "Running Autonomous Survival Cycle (300 ticks)...\n";
    std::cout << orch.execute_command("AUTONOMOUS_SURVIVAL_CYCLE 300") << "\n\n";

    std::cout << "Final Survival Status:\n" << orch.execute_command("FINANCE_STATUS") << "\n";
    return 0;
}
