#include "finance_orchestrator.hpp"
#include <iostream>
#include <string>

using namespace brain3::finance;

int main(int argc, char* argv[]) {
    FinanceOrchestrator orch(10000.0);

    bool json_stream = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json-stream" || arg == "-j") {
            json_stream = true;
        } else if (arg == "--status") {
            std::cout << orch.execute_command("FINANCE_STATUS") << std::endl;
            return 0;
        } else if (arg == "--sim") {
            std::cout << orch.execute_command("SIMULATE_MARKET_CYCLE BTC/USDT 100 0.0005 0.015") << std::endl;
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "THE BRAIN 3: Quantitative Finance Dedicated Branch with Survival Instinct\n"
                      << "Usage:\n"
                      << "  ./brain_finance --json-stream         # Interactive JSON-stream pipe\n"
                      << "  ./brain_finance --status              # Dump current survival instinct status\n"
                      << "  ./brain_finance --sim                 # Run 100-tick Monte Carlo simulation\n";
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
              << "========================================================================\n";
    std::cout << "Initial Survival State:\n" << orch.execute_command("FINANCE_STATUS") << "\n\n";

    std::cout << "Running Market Simulation (50 ticks)...\n";
    std::cout << orch.execute_command("SIMULATE_MARKET_CYCLE BTC/USDT 50 0.001 0.012") << "\n\n";

    std::cout << "Final Survival Status:\n" << orch.execute_command("FINANCE_STATUS") << "\n";
    return 0;
}
