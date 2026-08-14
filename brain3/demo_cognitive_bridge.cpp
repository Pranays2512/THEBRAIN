/*
 * demo_cognitive_bridge.cpp — Proves that the crisp engines now flow through
 * the fuzzy cognitive loop (SOM → Attention → WorkingMemory → EpisodicMemory).
 *
 * Without this bridge: solve("2x+4=10") never touches SOM, Emotion, or WM.
 * With this bridge:    solve("2x+4=10") lights up a SOM neuron, updates Emotion,
 *                      may store in EpisodicMemory if surprising, and writes the
 *                      result into BindingMemory for fuzzy recall.
 */
#include <iostream>
#include <iomanip>
#include "cognitive_bridge.hpp"

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: COGNITIVE BRIDGE DEMO\n";
    std::cout << "  Crisp Engines <-> SOM + Attention + WM + EpisodicMemory\n";
    std::cout << "=============================================================\n\n";

    // Construct the Brain (small grid for demo speed)
    // 32x32 SOM (1024 neurons), 64-dimensional concept space
    brain2::Brain brain(32, 32, 64,
        /*hidden_dim=*/128,
        /*wm_capacity=*/7,
        /*episodic_max=*/500,
        /*self_neurons=*/8,
        /*seed=*/42);

    std::cout << "[Brain] Constructed: " << brain.som.n_neurons
              << " SOM neurons, " << brain.n_dims << "D concept space\n\n";

    // Construct the bridge
    brain3::CognitiveBridge bridge(brain);

    // ── Experiment 1: Math Problem ────────────────────────────────────────────
    std::cout << "--- EXPERIMENT 1: Algebra Equation ---\n";
    brain3::engines::synthesis::Problem p1;
    p1.type     = "equation";
    p1.data_str = "2x + 4 = 10";
    p1.lhs      = "2*x + 4";
    p1.rhs      = "10";

    auto r1 = bridge.solve(p1);
    std::cout << "\n[Result] solved=" << r1.solved
              << " som_bmu=" << r1.som_bmu
              << " pred_err=" << std::fixed << std::setprecision(4) << r1.prediction_error
              << " episodic_stored=" << r1.episodic_stored << "\n";
    bridge.print_state();
    std::cout << "\n";

    // ── Experiment 2: Calculus Problem (different type → different SOM neuron) ─
    std::cout << "--- EXPERIMENT 2: Calculus Integration ---\n";
    brain3::engines::synthesis::Problem p2;
    p2.type     = "integrate";
    p2.data_str = "x^2";
    p2.expr_str = "x^2";

    auto r2 = bridge.solve(p2);
    std::cout << "\n[Result] solved=" << r2.solved
              << " som_bmu=" << r2.som_bmu
              << " pred_err=" << std::fixed << std::setprecision(4) << r2.prediction_error
              << " episodic_stored=" << r2.episodic_stored << "\n";
    bridge.print_state();
    std::cout << "\n";

    // ── Experiment 3: Same algebra again → BMU should be same or nearby ───────
    std::cout << "--- EXPERIMENT 3: Same algebra → SOM recognition ---\n";
    brain3::engines::synthesis::Problem p3;
    p3.type     = "equation";
    p3.data_str = "3x + 6 = 15";
    p3.lhs      = "3*x + 6";
    p3.rhs      = "15";

    auto r3 = bridge.solve(p3);
    std::cout << "\n[Result] solved=" << r3.solved
              << " som_bmu=" << r3.som_bmu
              << " (algebra BMU from exp1 was: " << r1.som_bmu << ")"
              << " — " << (r3.som_bmu == r1.som_bmu ? "SAME neuron! (SOM learned)" : "different neuron")
              << "\n";
    bridge.print_state();
    std::cout << "\n";

    // ── Experiment 4: Show EpisodicMemory has events ──────────────────────────
    std::cout << "--- EXPERIMENT 4: EpisodicMemory Check ---\n";
    int ep_count = brain.episodic.episode_count();
    std::cout << "[EpisodicMemory] Episodes stored: " << ep_count << "\n";
    if (ep_count > 0) {
        std::cout << "[EpisodicMemory] ✓ Brain remembered " << ep_count
                  << " surprising events from the solve cycle!\n";
    }

    // ── Experiment 5: Save everything to disk ────────────────────────────────
    std::cout << "\n--- EXPERIMENT 5: Save All Components to Disk ---\n";
    try {
        std::system("mkdir -p ./out/brain_save");
        bridge.save("./out/brain_save");
        std::cout << "[Save] ✓ SOM weights, Language, Predictor, Episodic, "
                     "Emotion, Intuition — all saved to ./out/brain_save/\n";
    } catch (const std::exception& e) {
        std::cout << "[Save] Warning: " << e.what() << "\n";
    }

    std::cout << "\n=============================================================\n";
    std::cout << "  DEMO COMPLETE\n";
    std::cout << "  The crisp engines are now fully wired into the cognitive loop.\n";
    std::cout << "  Every math/code/physics solve goes through:\n";
    std::cout << "    SOM → Attention → WorkingMemory → EpisodicMemory → Emotion\n";
    std::cout << "=============================================================\n";

    return 0;
}
