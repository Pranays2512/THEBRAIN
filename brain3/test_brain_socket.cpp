#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>

#include "crisp/engines/neural/brain_language_socket.hpp"
#include "crisp/engines/knowledge/knowledge_base.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"

using namespace brain2;
using namespace brain2::sockets;

void print_header(const std::string& title) {
    std::cout << "\n=========================================================================\n";
    std::cout << "🔌 " << title << "\n";
    std::cout << "=========================================================================\n";
}

int main() {
    print_header("THE BRAIN: MODULAR PLUG-AND-PLAY LANGUAGE & NEURAL SOCKET TEST");

    // 1. Initialize Knowledge Base and Reasoning Engine
    knowledge::KnowledgeBase kb;
    kb.add("socrates", "is_a", "human", "philosophy");
    kb.add("human", "is_a", "mortal", "philosophy");
    kb.add("turing", "invented", "computer", "cs");

    reasoning::ReasoningEngine reasoning;

    // 2. Initialize Brain Socket Hub
    BrainSocketHub hub(&kb, &reasoning);

    std::cout << "✅ Brain Socket Hub initialized successfully.\n";
    std::cout << "📋 Currently Plugged Sockets:\n";
    for (const auto& sock_id : hub.list_plugged_sockets()) {
        std::cout << "   - [" << sock_id << "]\n";
    }

    // =========================================================================
    // TEST 1: EXECUTE VIA PLUGGED STAMLAT SOCKET (Native Symplectic Neural)
    // =========================================================================
    print_header("TEST 1: EXECUTE VIA STAMLAT SOCKET (Pure Geometric Hamiltonian Flow)");
    hub.switch_active_socket("STAMLAT");
    std::cout << "Active Socket: [" << hub.get_active_socket() << "]\n";

    SocketRequest req1;
    req1.text_prompt = "socrates human mortal";
    req1.temperature = 0.0f; // Deterministic Strict Prover Mode
    req1.task_type = "INFERENCE";

    auto res1 = hub.execute(req1);
    std::cout << "  Status:        " << (res1.success ? "✅ SUCCESS" : "❌ FAILED") << "\n";
    std::cout << "  Socket ID:     " << res1.socket_id << "\n";
    std::cout << "  Output:        " << res1.output_text << "\n";
    std::cout << "  Energy Norm:   " << res1.energy_norm << "\n";
    std::cout << "  Latency:       " << res1.execution_time_us << " microseconds\n";
    std::cout << "  Proof Steps:\n";
    for (const auto& step : res1.proof_chain) {
        std::cout << "    * " << step << "\n";
    }
    assert(res1.success);
    assert(res1.socket_id == "STAMLAT");
    assert(res1.execution_time_us < 500.0); // Microsecond speed

    // =========================================================================
    // TEST 2: HOT-SWAP TO BrainQL SOCKET (Symbolic Declarative Query)
    // =========================================================================
    print_header("TEST 2: HOT-SWAP TO BrainQL SOCKET (Declarative Symbolic Logic)");
    bool swapped = hub.switch_active_socket("BrainQL");
    assert(swapped);
    std::cout << "Swapped Active Socket to: [" << hub.get_active_socket() << "]\n";

    SocketRequest req2;
    req2.text_prompt = "socrates is_a mortal";
    req2.task_type = "QUERY";

    auto res2 = hub.execute(req2);
    std::cout << "  Status:        " << (res2.success ? "✅ SUCCESS" : "❌ FAILED") << "\n";
    std::cout << "  Socket ID:     " << res2.socket_id << "\n";
    std::cout << "  Output:        " << res2.output_text << "\n";
    std::cout << "  Latency:       " << res2.execution_time_us << " microseconds\n";
    std::cout << "  Proof Steps:\n";
    for (const auto& step : res2.proof_chain) {
        std::cout << "    * " << step << "\n";
    }
    assert(res2.success);
    assert(res2.socket_id == "BrainQL");

    // =========================================================================
    // TEST 3: HOT-SWAP TO HYBRID DUAL MODE (STAMLAT + BrainQL Co-Supervision)
    // =========================================================================
    print_header("TEST 3: HYBRID DUAL MODE (STAMLAT Geometric + BrainQL Symbolic Consensus)");
    hub.switch_active_socket("HYBRID_DUAL");
    std::cout << "Active Socket: [" << hub.get_active_socket() << "]\n";

    SocketRequest req3;
    req3.text_prompt = "turing computer";
    req3.temperature = 0.1f; // Slight Langevin exploration
    req3.task_type = "INFERENCE";

    auto res3 = hub.execute(req3);
    std::cout << "  Status:        " << (res3.success ? "✅ SUCCESS" : "❌ FAILED") << "\n";
    std::cout << "  Socket ID:     " << res3.socket_id << "\n";
    std::cout << "  Dual Output:   " << res3.output_text << "\n";
    std::cout << "  Dual Latency:  " << res3.execution_time_us << " microseconds\n";
    std::cout << "  Metadata:      " << res3.metadata << "\n";
    std::cout << "  Combined Proof Trail:\n";
    for (const auto& step : res3.proof_chain) {
        std::cout << "    * " << step << "\n";
    }
    assert(res3.success);
    assert(res3.socket_id == "HYBRID_DUAL");

    // =========================================================================
    // TEST 4: DYNAMIC RUNTIME PLUGGING / UNPLUGGING
    // =========================================================================
    print_header("TEST 4: DYNAMIC RUNTIME UNPLUG & PLUG VERIFICATION");
    hub.unplug_socket("STAMLAT");
    std::cout << "Unplugged [STAMLAT]. Remaining Sockets:\n";
    for (const auto& sock_id : hub.list_plugged_sockets()) {
        std::cout << "   - [" << sock_id << "]\n";
    }
    assert(hub.list_plugged_sockets().size() == 1);

    // Re-plug new customized 32-dim STAMLAT plugin
    hub.plug_socket(std::make_shared<STAMLATSocketPlugin>(32, 8));
    std::cout << "Re-plugged high-capacity 32-dim STAMLAT engine. Sockets:\n";
    for (const auto& sock_id : hub.list_plugged_sockets()) {
        std::cout << "   - [" << sock_id << "]\n";
    }
    assert(hub.list_plugged_sockets().size() == 2);

    print_header("ALL PLUG-AND-PLAY SOCKET ARCHITECTURE TESTS COMPLETED SUCCESSFULLY");
    std::cout << "🎉 100% MODULARITY & HOT-SWAP VERIFIED (STAMLAT <-> BrainQL <-> HYBRID)\n\n";

    return 0;
}
